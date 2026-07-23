"""
Brute-force test: try all FMA patterns for weight decay on the midpoint values.
"""
import numpy as np
import struct

# The known value that produces midpoints
p_f32 = np.float32(0.01396983861923218)  # hex 3c64e1c0
p_d = np.float64(p_f32)
D = np.float64(1e-5) * np.float64(0.01)  # 1e-7

print(f"p = {p_f32:.15e} (0x{struct.pack('>f', p_f32).hex()})")
print(f"D = {D:.20e}")
print(f"p_d = {p_d:.20e}")

# The exact double result
exact = p_d * (1.0 - D)
print(f"\nexact p*(1-D) = {exact:.20e}")

# Round-to-nearest-even
val_f32 = np.float32(exact)
val_bits = np.array([val_f32], dtype=np.float32).view(np.uint32)[0]
val_up = np.array([val_bits + 1], dtype=np.uint32).view(np.float32)[0]

print(f"f32_round_down = {val_f32:.15e} (0x{val_bits:08x}) bit0={'even' if val_bits%2==0 else 'odd'}")
print(f"f32_round_up   = {val_up:.15e} (0x{val_bits+1:08x}) bit0={'even' if (val_bits+1)%2==0 else 'odd'}")

# Since it's a midpoint, RNE picks the even one
dist_down = abs(np.float64(val_f32) - exact)
dist_up = abs(np.float64(val_up) - exact)
print(f"dist_down = {dist_down:.5e}")
print(f"dist_up   = {dist_up:.5e}")
print(f"midpoint: {dist_down == dist_up}")

if val_bits % 2 == 0:
    print(f"RNE picks: round_down (0x{val_bits:08x}) -- even")
else:
    print(f"RNE picks: round_up (0x{val_bits+1:08x}) -- even")

# Now check: what does the NON-FMA path give?
# Non-FMA: temp = round(D * p_d), result = round(p_d - temp)
temp_exact = D * p_d
temp_f64 = np.float64(temp_exact)  # D*p is exact in double (both have limited mantissa bits)
print(f"\nD*p exact = {temp_exact:.20e}")

# But wait - in the non-FMA path, D*p is computed in double and is exact in double
# Then p_d - temp_f64 is also exact in double (since we haven't lost precision)
# So non-FMA double gives the same result as FMA double for this case

# The real question: is the weight decay computed in FLOAT or DOUBLE?
# In PyTorch: param -= lr * weight_decay * param
# lr (double) * weight_decay (double) = D (double)
# D * param (float -> promoted to double) = result (double)
# param -= result: param (float, promoted to double) - result (double) = double, truncated to float

# So the final truncation to float is where the midpoint rounding happens.
# Both FMA and non-FMA produce the same double result (since D*p and p-D*p are exact in double).
# The only thing that matters is: does the double->float conversion round to 0x3c64e1be or 0x3c64e1bf?

# Let's check what numpy (which uses RNE) gives:
print(f"\nnumpy RNE: 0x{val_bits:08x}")

# Now the question is: which way does PyTorch round?
# PyTorch output for this element goes to PT_final different from PD_final by 1 ULP at the END.
# The weight-decayed p is an intermediate. Let me check if the intermediate p_wd
# being off by 1 ULP would cause the final p to differ by 1 ULP too.

# Actually, the problem might be simpler: both frameworks get the SAME p_wd (since
# the midpoint rounding is deterministic with RNE). The 1 ULP diff at the final
# output comes from the subsequent p -= step_size * exp_avg / denom computation.

# Since the intermediate p_wd is at a midpoint, the specific float value it gets
# determines how p -= step_size * ... rounds. And if both frameworks get the SAME
# p_wd but a different final p, then the issue is in p -= step_size * exp_avg / denom.

# But wait - we already showed that with weight_decay=0, p -= ... gives 0 diffs.
# So the issue is that the weight-decayed p is different.

# HYPOTHESIS: the FMA changes the double result BEFORE truncation.
# fma(-D, p, p) = (-D)*p + p computed with infinite precision.
# (-D)*p + p = p*(1-D) exactly (no rounding needed, exact in infinite precision).
# So FMA(double) gives EXACTLY p*(1-D), same as D*p and then p-D*p.
# Therefore FMA vs non-FMA doesn't matter for double.

# The ONLY source of difference: the final truncation from double to float.
# This is deterministic (RNE), so both should give the same result...
# UNLESS one framework computes in FLOAT instead of DOUBLE!

# Check: what if param is NOT promoted to double?
# float -= double * double * float
# = float -= (double_D * float_param_promoted_to_double)
# = float -= double_result
# = float = float_promoted_to_double - double_result = double, truncated to float

# OR what if the compiler keeps it in float?
# float -= float(D) * float(param)
# D_f32 = float(1e-7) = which float32 value?
D_f32 = np.float32(D)
print(f"\nD as float32: {D_f32:.15e} (0x{struct.pack('>f', D_f32).hex()})")
print(f"D as float64: {D:.20e}")
print(f"D_f32 == D? {np.float64(D_f32) == D}")

# If computed in float:
temp_f32 = np.float32(np.float32(D_f32) * np.float32(p_f32))
p_wd_float = np.float32(np.float32(p_f32) - temp_f32)
print(f"\nFloat-only weight decay:")
print(f"  D_f32 * p_f32 = {temp_f32:.15e}")
print(f"  p - temp = {p_wd_float:.15e} (0x{struct.pack('>f', p_wd_float).hex()})")

# If computed in double:
temp_d = D * p_d
p_wd_double = np.float32(p_d - temp_d)
print(f"\nDouble weight decay:")
print(f"  D * p_d = {temp_d:.20e}")
print(f"  p_d - temp = {p_d - temp_d:.20e}")
print(f"  truncated to f32 = {p_wd_double:.15e} (0x{struct.pack('>f', p_wd_double).hex()})")

# Check: is D exactly representable as float64?
print(f"\nD = {D} = {D:.20e}")
print(f"Is D = 1e-7 exactly? {D == 1e-7}")
print(f"1e-7 bits: {struct.pack('>d', 1e-7).hex()}")
print(f"D bits:    {struct.pack('>d', D).hex()}")
# D = 1e-5 * 0.01 in double. Is this the same as 1e-7?
# 1e-5 * 0.01 has TWO roundings (1e-5 rounded, 0.01 rounded, product rounded)
# while 1e-7 has ONE rounding. They might differ!
D_direct = 1e-7
print(f"\n1e-7 direct: {D_direct:.20e} ({struct.pack('>d', D_direct).hex()})")
print(f"1e-5 * 0.01: {D:.20e} ({struct.pack('>d', D).hex()})")
print(f"Are they the same? {D == D_direct}")
