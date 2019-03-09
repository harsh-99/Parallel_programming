import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

num_steps = 72000
file = open('out.txt')
data = file.read().split('\n')

X = []
Y = []
Z = []
for row in data:
    if(row == ''):
        break
    if(row[0] == 'S'):
        continue
    x = []
    y = []
    z = []
    cols = row.split(' ')
    for i in range(0, len(cols), 3):
        if(i == 3000):
            break
        # print i
        x.append(cols[i])
        y.append(cols[i + 1])
        z.append(cols[i + 2])
    X.append(x)
    Y.append(y)
    Z.append(z)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

print(X.shape)

def Gen_RandLine(index, dims=2):
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, num_steps))
    lineData[0] = X[0:num_steps][:, index]
    lineData[1] = Y[0:num_steps][:, index]
    lineData[2] = Z[0:num_steps][:, index]
    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, num - 2 : num])
        line.set_3d_properties(data[2, num - 2 : num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = [Gen_RandLine(index, 3) for index in range(10)]

lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([0, 100.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 200.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 400.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=200, blit=False)

plt.show()