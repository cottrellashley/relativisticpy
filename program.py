from src.relativisticpy.tensor.gr_tensor import GrTensor

tensor = GrTensor("[[-A(r),0,0,0],[0,B(r),0,0],[0,0,r**2,0],[0,0,0,r**2*smp.sin(theta)**2]]", "_{a}_{b}", "[t, r, theta, phi]")

print(tensor)