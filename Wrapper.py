import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R_SciPy
import scipy.optimize as optimize
import random

from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from NonLinearTriangulation import *
from DisambiguateCameraPose import *
from PnPRANSAC import *
from NonLinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *



def getEuler(R):
    r = R_SciPy.from_matrix(R)
    return r.as_euler('xyz', degrees=False)


def read_feature_matches(file_path):
    """ Reads feature matches from the given text file and returns structured matches. """
    matches = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_features = int(lines[0].split(':')[1].strip())  # Read number of features

    for line in lines[1:]:  # Skip first line
        data = list(map(float, line.split()))
        num_matches = int(data[0])  # Number of matches for this feature

        # Extract reference feature point in the current image
        u_curr, v_curr = data[4], data[5]

        if (u_curr, v_curr) in matches.keys():
            for i in data[6:]:
                matches[(u_curr, v_curr)].append(i)
            # print(matches[(u_curr, v_curr)])
        else:
          # print(data[6:])
          matches[(u_curr, v_curr)] = data[6:]

    return matches


def plot_points_2d(points1, points2, R, t):

    K = np.array([[531.122155322710, 0 ,407.192550839899], [0, 531.541737503901, 313.308715048366], [0, 0, 1]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at origin
    C = np.reshape(t, (3, 1))
    I = np.identity(3)
    P2 = np.dot(K, np.dot(R, np.hstack((I, -C))))           # Second camera pose

    valid_count = 0
    X = []
    for x1, x2 in zip(points1, points2):
        arr = linear_triangulation(x1, x2, P1, P2)
        # print(arr)
        X.append([arr[0],arr[1],arr[2]])


    return np.array(X)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Outputs', default='../Outputs/', help='Directory to save output plots')
    parser.add_argument('--Data', default='../P2Data', help='Directory with matchingX.txt and images')
    args = parser.parse_args()
    os.makedirs(args.Outputs, exist_ok=True)
    data = args.Data
    outputs = args.Outputs


    # Extract matches for all combinations and store them in a dictionary
    matches_dict = {}

    image_1_matches = read_feature_matches(data+'/matching1.txt')
    ctr1 = 0
    ctr2 = 0
    ctr3 = 0

    for key, value in image_1_matches.items():
        for i in value:
            if i == 2:
                # if 2 in value:
                ctr3 += 1
                if "pts_1_2_1" not in matches_dict.keys():
                    ctr1 += 1
                    matches_dict["pts_1_2_1"] = []
                    matches_dict["pts_1_2_1"].append(key)
                    matches_dict["pts_1_2_2"] = []
                    matches_dict["pts_1_2_2"].append(value[value.index(2)+1:value.index(2)+3])
                else:
                    ctr2 += 1
                    matches_dict["pts_1_2_1"].append(key)
                    matches_dict["pts_1_2_2"].append(value[value.index(2)+1:value.index(2)+3])

            if i==3:
                if "pts_1_3_1" not in matches_dict.keys():
                    matches_dict["pts_1_3_1"] = []
                    matches_dict["pts_1_3_1"].append(key)
                    matches_dict["pts_1_3_2"] = []
                    matches_dict["pts_1_3_2"].append(value[value.index(3)+1:value.index(3)+3])
                else:
                    matches_dict["pts_1_3_1"].append(key)
                    matches_dict["pts_1_3_2"].append(value[value.index(3)+1:value.index(3)+3])

            if i==4:
                if "pts_1_4_1" not in matches_dict.keys():
                    matches_dict["pts_1_4_1"] = []
                    matches_dict["pts_1_4_1"].append(key)
                    matches_dict["pts_1_4_2"] = []
                    matches_dict["pts_1_4_2"].append(value[value.index(4)+1:value.index(4)+3])
                else:
                    matches_dict["pts_1_4_1"].append(key)
                    matches_dict["pts_1_4_2"].append(value[value.index(4)+1:value.index(4)+3])

            if i==5:
                if "pts_1_5_1" not in matches_dict.keys():
                    matches_dict["pts_1_5_1"] = []
                    matches_dict["pts_1_5_1"].append(key)
                    matches_dict["pts_1_5_2"] = []
                    matches_dict["pts_1_5_2"].append(value[value.index(5)+1:value.index(5)+3])

                else:
                    matches_dict["pts_1_5_1"].append(key)
                    matches_dict["pts_1_5_2"].append(value[value.index(5)+1:value.index(5)+3])


    image_2_matches = read_feature_matches(data+'/matching2.txt')

    for key, value in image_2_matches.items():
        for i in value:
            if i==3:
                if "pts_2_3_1" not in matches_dict.keys():
                    matches_dict["pts_2_3_1"] = []
                    matches_dict["pts_2_3_1"].append(key)
                    matches_dict["pts_2_3_2"] = []
                    matches_dict["pts_2_3_2"].append(value[value.index(3)+1:value.index(3)+3])
                else:
                    matches_dict["pts_2_3_1"].append(key)
                    matches_dict["pts_2_3_2"].append(value[value.index(3)+1:value.index(3)+3])

            if i==4:
                if "pts_2_4_1" not in matches_dict.keys():
                    matches_dict["pts_2_4_1"] = []
                    matches_dict["pts_2_4_1"].append(key)
                    matches_dict["pts_2_4_2"] = []
                    matches_dict["pts_2_4_2"].append(value[value.index(4)+1:value.index(4)+3])

                else:
                    matches_dict["pts_2_4_1"].append(key)
                    matches_dict["pts_2_4_2"].append(value[value.index(4)+1:value.index(4)+3])

            if i==5:
                if "pts_2_5_1" not in matches_dict.keys():
                    matches_dict["pts_2_5_1"] = []
                    matches_dict["pts_2_5_1"].append(key)
                    matches_dict["pts_2_5_2"] = []
                    matches_dict["pts_2_5_2"].append(value[value.index(5)+1:value.index(5)+3])

                else:
                    matches_dict["pts_2_5_1"].append(key)
                    matches_dict["pts_2_5_2"].append(value[value.index(5)+1:value.index(5)+3])


    image_3_matches = read_feature_matches(data+'/matching3.txt')

    for key, value in image_3_matches.items():
        for i in value:
            if i==4:
                if "pts_3_4_1" not in matches_dict.keys():
                    matches_dict["pts_3_4_1"] = []
                    matches_dict["pts_3_4_1"].append(key)
                    matches_dict["pts_3_4_2"] = []
                    matches_dict["pts_3_4_2"].append(value[value.index(4)+1:value.index(4)+3])

                else:
                    matches_dict["pts_3_4_1"].append(key)
                    matches_dict["pts_3_4_2"].append(value[value.index(4)+1:value.index(4)+3])

            if i==5:
                if "pts_3_5_1" not in matches_dict.keys():
                    matches_dict["pts_3_5_1"] = []
                    matches_dict["pts_3_5_1"].append(key)
                    matches_dict["pts_3_5_2"] = []
                    matches_dict["pts_3_5_2"].append(value[value.index(5)+1:value.index(5)+3])

                else:
                    matches_dict["pts_3_5_1"].append(key)
                    matches_dict["pts_3_5_2"].append(value[value.index(5)+1:value.index(5)+3])


    image_4_matches = read_feature_matches(data+'/matching4.txt')

    for key, value in image_4_matches.items():
        for i in value:
            if i==5:
                if "pts_4_5_1" not in matches_dict.keys():
                    matches_dict["pts_4_5_1"] = []
                    matches_dict["pts_4_5_1"].append(key)
                    matches_dict["pts_4_5_2"] = []
                    matches_dict["pts_4_5_2"].append(value[value.index(5)+1:value.index(5)+3])

                else:
                    matches_dict["pts_4_5_1"].append(key)
                    matches_dict["pts_4_5_2"].append(value[value.index(5)+1:value.index(5)+3])



    # Perform RANSAC on all feature matches
    print("Performing RANSAC and Fundamental Matrix calculation on feature matches .....")

    pts1 = np.array(matches_dict["pts_1_2_1"])
    pts2 = np.array(matches_dict["pts_1_2_2"])
    F_ransac_1_2, inlier_indices = ransac_fundamental_matrix(pts1, pts2)
    print("Refined Fundamental Matrix:\n", F_ransac_1_2)
    print("Number of inliers:", len(inlier_indices))
    pts1 = pts1[inlier_indices]
    pts2 = pts2[inlier_indices]
    matches_dict["pts_1_2_1"] = pts1
    matches_dict["pts_1_2_2"] = pts2

    points1 = np.array(matches_dict["pts_1_3_1"])
    points2 = np.array(matches_dict["pts_1_3_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_1_3_1"] = points1
    matches_dict["pts_1_3_2"] = points2

    points1 = np.array(matches_dict["pts_1_4_1"])
    points2 = np.array(matches_dict["pts_1_4_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_1_4_1"] = points1
    matches_dict["pts_1_4_2"] = points2

    points1 = np.array(matches_dict["pts_1_5_1"])
    points2 = np.array(matches_dict["pts_1_5_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_1_5_1"] = points1
    matches_dict["pts_1_5_2"] = points2

    points1 = np.array(matches_dict["pts_2_3_1"])
    points2 = np.array(matches_dict["pts_2_3_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_2_3_1"] = points1
    matches_dict["pts_2_3_2"] = points2

    points1 = np.array(matches_dict["pts_2_4_1"])
    points2 = np.array(matches_dict["pts_2_4_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_2_4_1"] = points1
    matches_dict["pts_2_4_2"] = points2

    points1 = np.array(matches_dict["pts_2_5_1"])
    points2 = np.array(matches_dict["pts_2_5_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_2_5_1"] = points1
    matches_dict["pts_2_5_2"] = points2

    points1 = np.array(matches_dict["pts_3_4_1"])
    points2 = np.array(matches_dict["pts_3_4_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_3_4_1"] = points1
    matches_dict["pts_3_4_2"] = points2

    points1 = np.array(matches_dict["pts_3_5_1"])
    points2 = np.array(matches_dict["pts_3_5_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_3_5_1"] = points1
    matches_dict["pts_3_5_2"] = points2

    points1 = np.array(matches_dict["pts_4_5_1"])
    points2 = np.array(matches_dict["pts_4_5_2"])
    F_ransac, inlier_indices = ransac_fundamental_matrix(points1, points2)
    points1 = points1[inlier_indices]
    points2 = points2[inlier_indices]
    matches_dict["pts_4_5_1"] = points1
    matches_dict["pts_4_5_2"] = points2


    # Compute Essential Matrix
    K = np.array([[531.122155322710, 0 ,407.192550839899], [0, 531.541737503901, 313.308715048366], [0, 0, 1]])
    E = compute_essential_matrix(F_ransac_1_2, K, K)
    print("Essential Matrix:\n", E)


    # Decompose Essential Matrix to get all possible camera poses
    possible_poses = decompose_essential_matrix(E)
    print("Possible Camera Poses:")
    all_centres = []
    rot_vectors = []
    for i, (R, t) in enumerate(possible_poses):
        print(f"Pose {i+1}:")
        print("Rotation Matrix:")
        print(R)
        print("Translation Vector:")
        print(t)
        rot_vectors.append(R)

        camera_centre = -R.T @ t
        all_centres.append(camera_centre)
        print("Camera Centre:")
        print(camera_centre)


    # Plot all possible 3D points from the camera poses using Linear Triangulation
    R1,t1 = possible_poses[0]
    X1 = plot_points_2d(pts1, pts2, R1, t1)
    R2,t2 = possible_poses[1]
    X2 = plot_points_2d(pts1, pts2, R2, t2)
    R3,t3 = possible_poses[2]
    X3 = plot_points_2d(pts1, pts2, R3, t3)
    R4,t4 = possible_poses[3]
    X4 = plot_points_2d(pts1, pts2, R4, t4)

    plt.figure(figsize=(8, 6))
    plt.scatter(X1[:, 0], X1[:, 2], c='b', label="Projected from Camera 1",s=1)
    plt.scatter(X2[:, 0], X2[:, 2], c='g', label="Projected from Camera 2",s=1)
    plt.scatter(X3[:, 0], X3[:, 2], c='r', label="Projected from Camera 3",s=1)
    plt.scatter(X4[:, 0], X4[:, 2], c='y', label="Projected from Camera 4",s=1)

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    # Labels and view
    plt.xlabel("X Axis")
    plt.ylabel("Z Axis")
    plt.title("Feature Points Projection on X-Z Plane")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(outputs+"all_camera_poses_3d_points.png")



    # Extract the correct camera pose and get 3D points from Triangulation
    print("Performing linear triangulation ....")
    R, t = select_best_pose(K, possible_poses,pts1,pts2)

    camera_poses = []
    camera_poses.append((R, t))

    print("Best Rotation R:\n", R)
    print("Best Translation t:\n", t)

    X_final = plot_points_2d(pts1, pts2, R, t)


    # Plot the correct camera pose and respective 3D triangulated points
    plt.figure(figsize=(8, 6))
    plt.scatter(X_final[:, 0], X_final[:, 2], c='b', label="Points",s=1)
    plt.xlim(-20, 20)
    plt.ylim(-5, 20)

    # Labels and view
    plt.xlabel("X Axis")
    plt.ylabel("Z Axis")
    plt.title("Projected 3D points using Linear Triangulation")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(outputs+"linear_triangulation_output_1_2.png")



    # Perform Non-Linear Triangulation to get optimized 3D points
    print("Performing non-linear triangulation .....")
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # First camera projection matrix (Identity)
    C = np.reshape(t, (3, 1))
    I = np.identity(3)
    P2 = np.dot(K, np.dot(R, np.hstack((I, -C))))

    X_optimized_list = []
    for i in range(pts1.shape[0]):
        X_init = X_final[i, :]  # Initial guess for the current point
        X_optimized = non_linear_triangulation(X_init, P1, P2, pts1[i, :], pts2[i, :])
        X_optimized_list.append(X_optimized)

    X_optimized = np.array(X_optimized_list)  # Convert list to array

    # print("Optimized 3D Points:\n", X_optimized)

    
    # Plot optimized points and the camera poses
    plt.figure(figsize=(8, 6))
    plt.scatter(X_final[:, 0], X_final[:, 2], c='r', label="linear triangulation",s=1)
    plt.scatter(X_optimized[:, 0], X_optimized[:, 2], c='b', label="non - linear triangulation",s=1)
    R_euler = getEuler(R)
    plt.plot(t[0], t[2],
            marker=(3, 0, int(np.degrees(R_euler[1]))),
            markersize=10, linestyle='None', label='Camera 1')
    R_euler = getEuler([[1,0,0],[0,1,0],[0,0,1]])
    plt.plot(0, 0,
            marker=(3, 0, int(np.degrees(R_euler[1]))),
            markersize=10, linestyle='None', label='Camera 2')
    plt.xlim(-40, 40)
    plt.ylim(-5, 40)

    # Labels and view
    plt.xlabel("X Axis")
    plt.ylabel("Z Axis")
    plt.title("Comparison of Projected 3D Points")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(outputs+"non_linear_triangulation_outputs_1_2.png")




    # Create a data structure to handle mapping between 3d points and images
    points_3d_dict = {}
    n = 3   # Number of images to consider

    for i in range(len(X_optimized)):
        x,y,z = X_optimized[i]
        points_3d_dict[(x,y,z)] = [(1,pts1[i][0],pts1[i][1]),(2,pts2[i][0],pts2[i][1])]


    

    # Loop through the next 3 image frames and construct the 3D points

    for image_index in range(3,n+1):

        # Prepare 3D-2D correspondences 
        x_next = []     # 2D correspondences
        X_req = []      # 3D correspondences

        if image_index==3 :
            for idx in range(len(pts1)):
                for i in range(len(matches_dict['pts_1_3_1'])):
                    if np.all(matches_dict['pts_1_3_1'][i] == pts1[idx]):
                        for j in range(len(matches_dict['pts_2_3_2'])):
                            if np.all(matches_dict['pts_1_3_2'][i] == matches_dict['pts_2_3_2'][j]):
                               
                                x_next.append(matches_dict['pts_2_3_2'][j])
                                X_req.append(X_optimized[idx])
                                points_3d_dict[(X_optimized[idx][0],X_optimized[idx][1],X_optimized[idx][2])].append((3, matches_dict['pts_2_3_2'][0], matches_dict['pts_2_3_2'][1]))

        elif image_index==4:
            common_points_4 = []
            common_points_4_indexes = []
            ptr = 0
            for point in matches_dict['pts_1_4_2']:
                if any(np.array_equal(point, p) for p in matches_dict['pts_2_4_2']) and any(np.array_equal(point, p) for p in matches_dict['pts_3_4_2']):
                    common_points_4.append(tuple(point))
                    common_points_4_indexes.append(ptr)
                    # common_points_4_indexes.append(matches_dict['pts_1_4_2'].index(point))
                ptr+= 1


            ctr = 0
            for key,value in points_3d_dict.items():
                index = common_points_4_indexes[ctr]
                if any((matches_dict['pts_1_4_1'][index][0], matches_dict['pts_1_4_1'][index][1]) == (v[1], v[2]) for v in value):
                    x_next.append(common_points_4[ctr])
                    X_req.append(key)


        x_next = np.array(x_next)
        X_req = np.array(X_req)

        # Perform Linear PnP to get camera pose
        print("Performing Linear PnP ....")
        R, t = PnP(X_req, x_next, K)
        print("Rotation Matrix R:\n", R)
        print("Translation Vector t:\n", t)

        # Plot the new camera pose w.r.t existing 3D points
        plt.figure(figsize=(6, 6))
        plt.title("Linear PnP output of Camera 3 Pose relative to current 3D points")
        plt.xlim(-20,20)
        plt.ylim(-5,20)
        R_euler = getEuler(R)
        plt.scatter(X_optimized[:, 0], X_optimized[:, 2], c='b', label="non - linear traiangulation",s=1)
        plt.plot(t[0], t[2],
                marker=(3, 0, int(np.degrees(R_euler[1]))),
                markersize=10, linestyle='None', label='Camera 3')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(outputs+"linear_pnp_outputs"+str(image_index)+".png")



        # Perform PnP RANSAC to get the correct camera pose
        print("Performing PnP RANSAC ....")
        best_C, best_R, best_inliers = ransac_pnp(X_req, x_next, K)
        print("Best Rotation Matrix R:\n", best_R)
        print("Best Translation Vector t:\n", best_C)

        # Plot the new pose after RANSAC w.r.t existing 3D points
        plt.figure(figsize=(6, 6))
        plt.title("PnP RANSAC output of Camera 3 Pose relative to current 3D points")
        plt.xlim(-20,20)
        plt.ylim(-5,20)
        R_euler = getEuler(best_R)
        plt.scatter(X_optimized[:, 0], X_optimized[:, 2], c='b', label="non - linear traiangulation",s=1)
        plt.plot(best_C[0], best_C[2],
                marker=(3, 0, int(np.degrees(R_euler[1]))),
                markersize=10, linestyle='None', label='Camera 3')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(outputs+"ransac_pnp_outputs"+str(image_index)+".png")       



        # Perform Non-Linear PnP 
        print("Performing Non-Linear PnP ....")
        R_opt, t_opt = NonLinearPnP(K, x_next, X_req, best_R, best_C)
        print("Rotation Matrix R:\n", R_opt)
        print("Translation Vector t:\n", t_opt)

        # Plot the new pose after non-linear PnP w.r.t existing 3D points 
        plt.figure(figsize=(6, 6))
        plt.title("Non -Linear PnP output of Camera 3 Pose relative to current 3D points")
        plt.xlim(-20,20)
        plt.ylim(-5,20)
        R_euler = getEuler(R_opt)
        plt.scatter(X_optimized[:, 0], X_optimized[:, 2], c='b', label="non - linear traiangulation",s=1)
        plt.plot(t_opt[0], t_opt[2],
                marker=(3, 0, int(np.degrees(R_euler[1]))),
                markersize=10, linestyle='None', label='Camera 3') 
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(outputs+"non_linear_pnp_outputs"+str(image_index)+".png")



        # Add this new camera pose to the list of camera poses
        camera_poses.append((R_opt, t_opt))



        # Get new set of 3D points using matches between camera 1 and 3 pose through linear triangulation
        points_2d_1 = matches_dict['pts_1_3_1']
        points_2d_2 = matches_dict['pts_1_3_2']

        X_new = plot_points_2d(points_2d_1, points_2d_2, R_opt, t_opt)

        # Plot triangulated points
        plt.figure(figsize=(8, 6))
        plt.scatter(X_new[:, 0], X_new[:, 2], c='b', label="Projected from Camera 1 and 3",s=1)
        plt.xlim(-20, 20)
        plt.ylim(-5, 20)
        plt.xlabel("X Axis")
        plt.ylabel("Z Axis")
        plt.title("Projected 3D Points by Triangulation between frames 1 and 3")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(outputs+"linear_triangulation"+str(image_index)+".png")




        # Perform non-linear triangulation to obtain optimized 3D points
        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # First camera projection matrix (Identity)
        C = np.reshape(t_opt, (3, 1))
        I = np.identity(3)
        P2 = np.dot(K, np.dot(R_opt, np.hstack((I, -C))))

        X_opt_list = []
        for i in range(points_2d_1.shape[0]):
            X_init = X_new[i, :]  # Initial guess for the current point
            X_opt = non_linear_triangulation(X_init, P1, P2, points_2d_1[i, :], points_2d_2[i, :])
            X_opt_list.append(X_opt)

        X_opt = np.array(X_opt_list)  # Convert list to array


        # Plot the optimized 3D points
        plt.figure(figsize=(8, 6))
        plt.scatter(X_new[:, 0], X_new[:, 2], c='b', label="linear triangulation",s=1)
        plt.scatter(X_opt[:, 0], X_opt[:, 2], c='r', label="non linear triangulation",s=1)
        plt.xlim(-20, 20)
        plt.ylim(-5, 20)
        plt.xlabel("X Axis")
        plt.ylabel("Z Axis")
        plt.title("Comparison of Projected 3D Points between frames 1 and 3")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(outputs+"non_linear_triangulation"+str(image_index)+".png")
        



        # Construct the new 3D points
        extra_3d_points = []
        # print(points_2d_2)
        # print(x_next)

        present_indices = []
        for i in range(len(points_2d_2)):
            for j in range(len(x_next)):
                if np.all(points_2d_2[i] == x_next[j]):
                    present_indices.append(i)
                    break
                    
        # print(present_indices)
        count = 0
        indices_impt = []
        x_visible_1_3 = []
        x_visible_2_3 = []
        elems_frame_2 = []
        pts_3 = []
        pts_3_rem = []

        for i in range(len(points_2d_1)):
            if i not in present_indices:
                extra_3d_points.append(X_opt[i])
                x_visible_1_3.append([points_2d_1[i][0],points_2d_1[i][1]])

                for elem in matches_dict['pts_2_3_2']:
                    if np.all(elem == points_2d_2[i]):
                        indices_impt.append(count)
                        elems_frame_2.append(elem)
                        break
                count+=1

        # print("indices:",indices_impt)

        ptr = 0

        for k in range(len(extra_3d_points)):
            if k in indices_impt:
                x_visible_2_3.append([elems_frame_2[ptr][0],elems_frame_2[ptr][1]])
                ptr+=1
            else:
                x_visible_2_3.append([0,0])


        for m in range(X_optimized.shape[0]):
            if m in present_indices:
                pts_3.append(points_2d_2[m])
            else:
                pts_3.append([0,0])


        for i in range(len(points_2d_1)):
            if i not in present_indices:
                pts_3_rem.append(points_2d_2[i])


        # print(len(pts_3))
        # print(len(pts_3_rem))


        extra_3d_points = np.array(extra_3d_points)
        # print(extra_3d_points.shape)
        # print(X_opt.shape)
        # print(X_optimized.shape)

        X_final_merged = np.concatenate((X_optimized, extra_3d_points), axis=0)


        print("Getting Visibility Matrix and performing Bundle Adjustment ....")

        # Obtain visibility matrix
        V = getVisibilityMatrix(X_final_merged,X_optimized,indices_impt,present_indices)
        print(f"Visibility matrix : {V}")

        # Perform Bundle Adjustment
        observed_2d = np.random.rand(3, X_final_merged.shape[0], 2)  # (num_cameras, num_points, 2)

        observed_2d[0,:X_optimized.shape[0]] = pts1
        observed_2d[0,X_optimized.shape[0]:] = x_visible_1_3
        observed_2d[1,:X_optimized.shape[0]] = pts2
        observed_2d[1,X_optimized.shape[0]:] = x_visible_2_3
        observed_2d[2,:X_optimized.shape[0]] = pts_3
        observed_2d[2,X_optimized.shape[0]:] = pts_3_rem

        optimized_cameras, optimized_X = bundle_adjustment(camera_poses, X_final_merged, K, V, observed_2d)
        camera_poses = optimized_cameras

        # Print results
        print("\nOptimized Camera Poses:")
        for i, (R_opt, C_opt) in enumerate(optimized_cameras):
            print(f"Camera {i+1}:\nRotation:\n{R_opt}\nCenter:\n{C_opt}\n")

        # Plot the final camera poses and 3D points
        plt.figure(figsize=(6, 6))
        plt.title("Final Bundle Adjusted Outputs")
        plt.xlim(-20,20)
        plt.ylim(-5,20)

        plt.scatter(optimized_X[:, 0], optimized_X[:, 2], c='b', label="Projected 3D Points",s=1)
        R_euler_1 = getEuler([[1,0,0],[0,1,0],[0,0,1]])
        # t_opt_1 = optimized_cameras[0][1]
        R_euler_2 = getEuler(optimized_cameras[0][0])
        t_opt_2 = optimized_cameras[0][1]
        R_euler_3 = getEuler(optimized_cameras[1][0])
        t_opt_3 = optimized_cameras[1][1]
        


        plt.plot(0, 0,
                marker=(3, 0, int(np.degrees(R_euler_1[1]))),
                markersize=10, linestyle='None', label='Camera 1')
        plt.plot(t_opt_2[0], t_opt_2[2],
                marker=(3, 0, int(np.degrees(R_euler_2[1]))),
                markersize=10, linestyle='None', label='Camera 2')
        plt.plot(t_opt_3[0], t_opt_3[2],
                marker=(3, 0, int(np.degrees(R_euler_3[1]))),
                markersize=10, linestyle='None', label='Camera 3')
        
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(outputs+"final_output_after_bundle_adjustment"+str(image_index)+".png")




if __name__ == '__main__':
    main()
