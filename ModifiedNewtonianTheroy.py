import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Shape:
    '''Shape creator
        - Creates the shape of the payload
        - Returns the coordinate list of the payload and ready for plotting
        
        Origin at the nose cone tip.
        Angles in degrees, lengths in centimetres.
    '''
    nose_angle        = 7
    nose_length       = 111.8

    body_length       = 161.8 - 111.8

    flare_wedge_angle  = 33
    flare_wedge_length = 168.04 - 161.8
    flare_length       = 172.17 - 168.04


class CoordPlotter(Shape):
    '''CoordPlotter
        - Computes the 2D profile coordinates of the payload outline.
        - Returns list_x (axial) and list_y (radial) as flat Python lists.
    '''

    @staticmethod
    def draw():
        list_x = []
        list_y = []

        x, y = 0.0, 0.0

        # Origin — nose tip
        list_x.append(x)
        list_y.append(y)

        # Nose cone
        x += Shape.nose_length
        y += Shape.nose_length * np.tan(np.deg2rad(Shape.nose_angle))
        list_x.append(x)
        list_y.append(y)

        # Cylindrical body (y stays constant)
        x += Shape.body_length
        list_x.append(x)
        list_y.append(y)

        # Flare wedge (expanding section)
        x += Shape.flare_wedge_length
        y += Shape.flare_wedge_length * np.tan(np.deg2rad(Shape.flare_wedge_angle))
        list_x.append(x)
        list_y.append(y)

        # Flare extrusion (y stays constant)
        x += Shape.flare_length
        list_x.append(x)
        list_y.append(y)

        # # Mirror the rocket
        # list_x += list_x[::-1]
        # list_y += [-y for y in list_y[::-1]]
        

        return np.array([
            np.array(list_x, dtype=np.float64),
            np.array(list_y, dtype=np.float64)
            ])
        
    @staticmethod
    def interpolate(list_x, list_y, n_points=None):
        """
        Interpolates n_points between each consecutive coordinate pair.
        """
        interp_x = []
        interp_y = []

        for i in range(len(list_x) - 1):
            x_seg = np.linspace(list_x[i], list_x[i+1], n_points, endpoint=False)
            y_seg = np.linspace(list_y[i], list_y[i+1], n_points, endpoint=False)
            interp_x.append(x_seg)
            interp_y.append(y_seg)

        # Append the final point
        interp_x.append([list_x[-1]])
        interp_y.append([list_y[-1]])

        return np.array([
            np.concatenate(interp_x),
            np.concatenate(interp_y)
        ])
    


# MODIFIED NEWTONIAN THEORY
class MNT:
    '''Modified Newtonian Theory
        - Takes interpolated coordinate array
        - Computes pressure coefficients along the surface
    '''


    def __init__(self, coords, aoa, M):
        self.coords = coords
        self.x = coords[0, :]
        self.y = coords[1, :]
        self.aoa = aoa
        self.M = M

    def print_coords(self):
        print("X:", self.x)
        print("Y:", self.y)

    

    def coefficient_of_pressure(self):
        panel_vector = np.vstack([
            np.diff(self.x),
            np.diff(self.y)
        ])
        print("PANEL:", panel_vector)

        n_segments = len(self.x) - 1  # one vector per segment

        freestream_vector = np.array([
            np.full(n_segments, np.cos(np.deg2rad(self.aoa))),  # x-component
            np.full(n_segments, np.sin(np.deg2rad(self.aoa)))   # y-component
        ])  # shape: (2, n_segments)

        # Cp_max via Pitot
        gamma = 1.4
        Cp_max = (2 / (gamma * self.M**2)) * (
            ((( (gamma+1)**2 * self.M**2 ) / (4*gamma*self.M**2 - 2*(gamma-1)))**(gamma/(gamma-1))) *
            ((1 - gamma + 2*gamma*self.M**2) / (gamma+1)) - 1
        )

        # Dot product to get cos(theta), then flow incidence angle beta
        Cp = np.zeros(n_segments)  # 1D, one value per segment

        for i in range(n_segments):
            cos_theta = (
                np.dot(panel_vector[:, i], freestream_vector[:, i]) /
                (np.linalg.norm(panel_vector[:, i]) * np.linalg.norm(freestream_vector[:, i]))
            )
            Cp[i] = 2 * np.sin(np.arccos(cos_theta))**2 * Cp_max

        return Cp

    def panel_coords(self):
        return np.array([
            self.x[:-1] + np.diff(self.x)/2,
            self.y[:-1] + np.diff(self.y)/2,
        ])

class VanDeyek:
    print("Hello World")

# Running
plotter = CoordPlotter()
coords = plotter.draw()

coords = plotter.interpolate(coords[0, :], coords[1, :], 2)
n_coords = coords.shape[1]
print(n_coords)

# Freestream
aoa = 15
M = 6

# Modified Newtonian Theory
mnt = MNT(coords, aoa, M)
mnt.print_coords()
Cp = mnt.coefficient_of_pressure()
panel_coords = mnt.panel_coords()

# Final values of Cp, please verify
print(Cp)








# Figure Plotting
plt.figure()
# Payload geometry
plt.plot(coords[0, :], coords[1, :])
# Free stream values
n = 3
origins_y = [-10, 0, 10]
plt.quiver(
    np.full(n, -30),
    origins_y,
    np.full(n, np.cos(np.deg2rad(aoa))),
    np.full(n, np.sin(np.deg2rad(aoa))),
    scale=10,
    color='steelblue'
)

# Plot Properties
plt.title("Payload Profile")
plt.xlabel("Axial length (cm)")
plt.ylabel("Radial offset (cm)")
plt.axis('equal')
plt.grid(True)
# for i in range(0, n_coords-1):
#     plt.text(coords[0, i], coords[1, i], Cp[i])
# plt.show()


# Saving data into the File.txt
df = pd.DataFrame({
    "X": panel_coords[0, :],
    "Y": panel_coords[1, :],
    "Cp": Cp,
    
})
print(df)
df.to_csv(
    "File.txt",
    index=False,        # no row numbers
    sep=',',           # tab separated instead of comma (cleaner for numeric data)
    float_format='%.4f', # 4 decimal places on all floats
    header=True,        # keep column headers
)