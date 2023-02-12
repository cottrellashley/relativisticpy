
# Another Option would be to pip install mongo db and then simply initiate an instance of the database whilst user is running application

class MathJSONStore:

    def __init__(self):
        self.state = {
            "Schwarzschild" : "[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "SchildGeneral" : "[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "AntiDeSitter" : "[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "PolarCoordinates" : "[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "Minkowski" : "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
            "WeylLewisPapapetrou": "[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "ReissnerNordström" : "[[-(1 - (G)/(r) + (Q**2)/(r**2)),0,0,0],[0,1/(1 - (G)/(r) + (Q**2)/(r**2)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"
        }
