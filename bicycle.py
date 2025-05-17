import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
import scipy.integrate as integ

"""
Kinematic Bicycle Model

* car is modeled as a front and back wheel only
* car is modeled as RWD (FWD is also possible, but equations change slightly,
  if you want I can look into adding it)
* wheels do not slip
* computed positions I believe are for the position of the back wheel (knowing
  the wheelbase you could compute front wheel position or center of car)

References:

I liked this one a lot
https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html

more in-depth
https://scholarships.engin.umich.edu/wp-content/uploads/sites/36/2020/02/Final-Report.pdf

similar to the first, looked good though
https://www.shuffleai.blog/blog/Simple_Understanding_of_Kinematic_Bicycle_Model.html
"""

################################################################

# dummy example data
ntimes = 50
times = np.arange(ntimes, dtype=np.float64)  # sec
steering_wheel_angles = np.linspace(-200.0, 200.0, ntimes)  # deg
steering_ratio = 12
speed = np.ones(ntimes) * 25.0  # mph
wheelbase = 95.0  # inches

# initial position of car
x0 = 0.0
y0 = 0.0

# angle of the car relative to the +x axis at startup
initial_orientation_angle = 0.0

################################################################

# standardize units
speed *= sc.mph  # now in m/s
steering_wheel_angles = np.radians(steering_wheel_angles)
wheelbase *= sc.inch  # now in meters

# convert steering wheel angles to angles of the wheels relative to
#   the car orientation
wheel_steer_angles = steering_wheel_angles / steering_ratio

# compute the angular velocity of the car
angular_velocity = speed * np.tan(wheel_steer_angles) / wheelbase

# integrate the angular velocity of the car over time to get the car's
#   orientation
orientation = integ.cumulative_simpson(angular_velocity, x=times,
                                       initial=initial_orientation_angle)

# compute x and y velocities
dxdt = speed * np.cos(orientation)
dydt = speed * np.sin(orientation)

# integrate to get position
x = integ.cumulative_simpson(dxdt, x=times, initial=x0)
y = integ.cumulative_simpson(dydt, x=times, initial=y0)

plt.plot(x, y, "-ro")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
