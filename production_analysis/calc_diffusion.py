# ------------------------------------------------- #
# This script calculates the diffusion of the       #
# eigenvalues from a suite of RAMSES cr simulations #
# ------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# consitent plotting style
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.minor.size"] = 4.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.minor.size"] = 4.5
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"text.usetex": True})

# parse in the data path and plot save name extension
import argparse

parser = argparse.ArgumentParser(description="Plot the diffusion of the eigenvalues")
parser.add_argument("-d", type=str, help="path to the data file")
parser.add_argument("-n", type=str, help="name to save the plot as")

args = parser.parse_args()
data_path = (
    "/Users/mattsampson/Research/Teyssier/MattRamses/production_analysis/256/" + args.d
)
filename = data_path + "/global_quantities.txt"
save_name = args.n

# read in the text file
data = pd.read_csv(filename)
df = pd.DataFrame(data)
code_time = 1.1563e15
df = df[(df["time"] > 1.5 * code_time) & (df["time"] < 3.5 * code_time)]
df["tau"] = (df["time"] / 1.1563e15) * 3

# now take the derrivitive of the lambda values with respect to time and plot
diff1 = np.diff(df["lambda1"]) / np.diff(df["time"])
diff2 = np.diff(df["lambda2"]) / np.diff(df["time"])
diff3 = np.diff(df["lambda3"]) / np.diff(df["time"])

# ------------------------------------------- #
# now combine the data between the injections #
# ------------------------------------------- #
data1 = zip(
    df["tau"][1:],
    diff1,
    diff2,
    diff3,
    df["lambda1"][1:],
    df["lambda2"][1:],
    df["lambda3"][1:],
)
t_inj1 = []
t_inj2 = []
t_inj3 = []
diffx1 = []
diffx2 = []
diffx3 = []
diffy1 = []
diffy2 = []
diffy3 = []
diffz1 = []
diffz2 = []
diffz3 = []
x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []
z1 = []
z2 = []
z3 = []

for i, j, k, l, m, n, o in data1:
    if i > 5.7:
        t_inj3.append(i - 5.7)
        diffx3.append(j)
        diffy3.append(k)
        diffz3.append(l)
        x3.append(m)
        y3.append(n)
        z3.append(o)
    elif i > 5.1 and i < 5.7:
        t_inj2.append(i - 5.1)
        diffx2.append(j)
        diffy2.append(k)
        diffz2.append(l)
        x2.append(m)
        y2.append(n)
        z2.append(o)
    elif i > 4.5 and i < 5.1:
        t_inj1.append(i - 4.5)
        diffx1.append(j)
        diffy1.append(k)
        diffz1.append(l)
        x1.append(m)
        y1.append(n)
        z1.append(o)
    else:
        continue

# now take the averages
all_times = np.unique(np.concatenate((t_inj1, t_inj2, t_inj3)))

# Interpolate values for each vector at the combined times
interpolated_x1 = np.interp(all_times, t_inj1, diffx1)
interpolated_x2 = np.interp(all_times, t_inj2, diffx2)
interpolated_x3 = np.interp(all_times, t_inj3, diffx3)

interpolated_y1 = np.interp(all_times, t_inj1, diffy1)
interpolated_y2 = np.interp(all_times, t_inj2, diffy2)
interpolated_y3 = np.interp(all_times, t_inj3, diffy3)

interpolated_z1 = np.interp(all_times, t_inj1, diffz1)
interpolated_z2 = np.interp(all_times, t_inj2, diffz2)
interpolated_z3 = np.interp(all_times, t_inj3, diffz3)

# Interpolate values for each vector at the combined times
i_x1 = np.interp(all_times, t_inj1, x1)
i_x2 = np.interp(all_times, t_inj2, x2)
i_x3 = np.interp(all_times, t_inj3, x3)

i_y1 = np.interp(all_times, t_inj1, y1)
i_y2 = np.interp(all_times, t_inj2, y2)
i_y3 = np.interp(all_times, t_inj3, y3)

i_z1 = np.interp(all_times, t_inj1, z1)
i_z2 = np.interp(all_times, t_inj2, z2)
i_z3 = np.interp(all_times, t_inj3, z3)

# Stack interpolated values for convenience
interpolated_values_x = np.vstack((interpolated_x1, interpolated_x2, interpolated_x3))
interpolated_values_y = np.vstack((interpolated_y1, interpolated_y2, interpolated_y3))
interpolated_values_z = np.vstack((interpolated_z1, interpolated_z2, interpolated_z3))

# Stack interpolated values for convenience
i_values_x = np.vstack((i_x1, i_x2, i_x3))
i_values_y = np.vstack((i_y1, i_y2, i_y3))
i_values_z = np.vstack((i_z1, i_z2, i_z3))

# --------------------------------------------------------- #
# Calculate the average and standard deviation at each time #
# --------------------------------------------------------- #
ave_diff_x = np.mean(interpolated_values_x, axis=0)
std_x = np.std(interpolated_values_x, axis=0)
ave_diff_y = np.mean(interpolated_values_y, axis=0)
std_y = np.std(interpolated_values_y, axis=0)
ave_diff_z = np.mean(interpolated_values_z, axis=0)
std_z = np.std(interpolated_values_z, axis=0)

# Calculate the average and standard deviation at each time
ave_x = np.mean(i_values_x, axis=0)
st_x = np.std(i_values_x, axis=0)
ave_y = np.mean(i_values_y, axis=0)
st_y = np.std(i_values_y, axis=0)
ave_z = np.mean(i_values_z, axis=0)
st_z = np.std(i_values_z, axis=0)

# ------------------------------------------- #
# averaged plots of diffusion and eigenvalues #
# ------------------------------------------- #
# plot params
col_x = "navy"
col_y = "darkorchid"
col_z = "gray"
col_sub = "firebrick"
col_diff = "black"
f_size = 20
m_size = 8
if save_name == "MA2":
    lab = r"$\mathcal{M}_A=2$"
elif save_name == "MA10":
    lab = r"$\mathcal{M}_A=10$"
elif save_name == "MA05":
    lab = r"$\mathcal{M}_A=0.5$"

# diffusion plot
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.plot(all_times, ave_diff_x, color=col_x, lw=2)
plt.fill_between(
    all_times,
    ave_diff_x - std_x,
    ave_diff_x + std_x,
    color=col_x,
    alpha=0.2,
)
plt.ylabel(r"$\frac{d\lambda_1}{dt}$", fontsize=f_size)
plt.ylim([1e18, 7e24])
plt.axhline(1e24, color=col_sub, ls="--", label="sub-grid diffusion")
# plt.axvline(diff_t, color=col_diff, ls="--", label='diffusion time limit')
plt.xlim([0, diff_t])
plt.text(
    0.03,
    0.92,
    lab,
    fontsize=20,
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    color="black",
)
plt.legend(frameon=False, fontsize=15, loc="upper right")

plt.subplot(3, 1, 2)
plt.plot(all_times, ave_diff_y, label=r"$\lambda_2$", color=col_y, lw=2)
plt.fill_between(
    all_times,
    ave_diff_y - std_y,
    ave_diff_y + std_y,
    color=col_y,
    alpha=0.2,
)
plt.xlabel(r"$\tau$", fontsize=f_size)
plt.ylabel(r"$\frac{d\lambda_2}{dt}$", fontsize=f_size)
plt.ylim([1e18, 7e24])
plt.axhline(1e24, color=col_sub, ls="--", label="sub-grid diffusion")
plt.axvline(diff_t, color=col_diff, ls="--", label="diffusion time limit")
plt.xlim([0, diff_t])

plt.subplot(3, 1, 3)
plt.plot(all_times, ave_diff_z, label=r"$\lambda_3$", color=col_z, lw=2)
plt.fill_between(
    all_times,
    ave_diff_z - std_z,
    ave_diff_z + std_z,
    color=col_z,
    alpha=0.2,
)
plt.xlabel(r"$\tau$", fontsize=f_size)
plt.ylabel(r"$\frac{d\lambda_3}{dt}$", fontsize=f_size)
plt.ylim([1e18, 7e24])
plt.axhline(1e24, color=col_sub, ls="--", label="sub-grid diffusion")
# plt.axvline(diff_t, color=col_diff, ls="--", label='diffusion time limit')
plt.xlim([0, diff_t])
save_n = save_name + "_averaged.png"
plt.savefig(save_n, dpi=300)

# eigenvalue plot
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.plot(all_times, ave_x, color=col_x, lw=2)
plt.fill_between(
    all_times,
    ave_x - st_x,
    ave_x + st_x,
    color=col_x,
    alpha=0.2,
)
plt.ylabel(r"$\lambda_1$", fontsize=f_size)
plt.xlim([0, diff_t])
plt.legend(frameon=False, fontsize=15, loc="upper right")
plt.text(
    0.03,
    0.92,
    lab,
    fontsize=20,
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    color="black",
)

plt.subplot(3, 1, 2)
plt.plot(all_times, ave_y, label=r"$\lambda_2$", color=col_y, lw=2)
plt.fill_between(
    all_times,
    ave_y - st_y,
    ave_y + st_y,
    color=col_y,
    alpha=0.2,
)
plt.xlabel(r"$\tau$", fontsize=f_size)
plt.xlim([0, diff_t])
plt.ylabel(r"$\lambda_2$", fontsize=f_size)

plt.subplot(3, 1, 3)
plt.plot(all_times, ave_z, label=r"$\lambda_3$", color=col_z, lw=2)
plt.fill_between(
    all_times,
    ave_z - st_z,
    ave_z + st_z,
    color=col_z,
    alpha=0.2,
)
plt.xlabel(r"$\tau$", fontsize=f_size)
plt.ylabel(r"$\lambda_3$", fontsize=f_size)
plt.xlim([0, diff_t])
save_n = save_name + "_cov_averaged.png"
plt.savefig(save_n, dpi=300)
