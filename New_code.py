import cv2
# from PIL import Image
# import PIL
import numpy as np
import math


capture = cv2.VideoCapture('umcp.mpg')
ret, frame = capture.read()
pix = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
w, h = pix.shape

video1 = cv2.VideoWriter('fg81.avi', -1, 10, (352, 240))
video2 = cv2.VideoWriter('bg81.avi', -1, 10, (352, 240))

# Initialise Foreground and background frame
fg = np.zeros((w, h, 3), dtype=np.uint8, order='C')
bg = np.zeros((w, h, 3), dtype=np.uint8, order='C')

# Parameters
T = 0.2  # Proportion of data to be accounted for Background
alpha = 0.025  # Learning Constant
rho = 0.1  # Second Learning Rate

# For each pixel, we have taken K=3 Gaussian distributions (assuming independence and same variance of each color)

# Mean of each Gaussian (e.g. for each pixel [Mean1, Mean2, Mean3] = [R,G,B] representation)
mean1 = np.zeros((w, h, 3), dtype=np.float64, order='C')
mean2 = np.zeros((w, h, 3), dtype=np.float64, order='C')
mean3 = np.zeros((w, h, 3), dtype=np.float64, order='C')

# Variance of each Gaussian
cov = np.ones((w, h, 3), dtype=np.float64, order='C')

# Weight of each Gaussian, each value is Initialised to one.
weight = np.ones((w, h, 3), dtype=np.float64, order='C')

count = 0
while count <= 997:
    ret, frame = capture.read()
    # pix = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for x in range(w):
        for y in range(h):
            # Initialisation of Parameters and Distribution Values
            if count == 0:
                mean1[x][y][:] = [40, 90, 160]
                mean2[x][y][:] = [40, 90, 160]
                mean3[x][y][:] = [40, 90, 160]
                cov[x][y][:] = [20, 20, 20]
                weight[x][y][:] = [0.33, 0.33, 0.34]

            dist = np.zeros((2, 3), dtype=np.float64, order='C')
            match = -1

            # Approximate K-means algorithm, In each iteration -
            # We check for the matching Distribution and then update the parameters of the matched one
            # And keep the others same

            for i in range(0, 3):

                temp = frame[x][y][:] - [mean1[x][y][i], mean2[x][y][i], mean3[x][y][i]]

                # Following is the square of the Mahalanobis Distance (the exponent term in Gaussian Distribution)
                
                euclid_distance = np.dot(temp.transpose(), temp)
                threshold = (1 / cov[x][y][i]) * euclid_distance

                # If the threshold is less than (2.5*2.5) = 6.25, we consider it as a matched distribution
                
                if threshold < (2.5 * 2.5):

                    match = i

                    # rho = alpha * gauss(m[x][y][match], v[x][y][match], value)

                    # updating the mean of the matched distribution
                    mean1[x][y][i] = (1 - rho) * mean1[x][y][i] + rho * frame[x][y][0]
                    mean2[x][y][i] = (1 - rho) * mean2[x][y][i] + rho * frame[x][y][1]
                    mean3[x][y][i] = (1 - rho) * mean3[x][y][i] + rho * frame[x][y][2]

                    # updating the variance of the matched class
                    cov[x][y][i] = (1 - rho) * cov[x][y][i] + (rho * euclid_distance)

                    # updating the weight of the matched class
                    weight[x][y][i] = (1 - alpha) * weight[x][y][i] + alpha



                else:

                    # updating the weight of the non-matched class
                    weight[x][y][i] = (1 - alpha) * weight[x][y][i]

                    # NOTE-sum of the weights after updating is equal to one, if the sum was one before update. Hence, no normalization required
                # -----------------------------------------------------------------------

            # Maintain the distributions in decreasing order of Weight/(sqrt(Variance))

            # 'dist' is a 2x3 array, first row to store the index of the distributions
            # and second row to store the w/sigma ratio of the distributions
            # 'dist[1][:]' array is stored with the W/sigma ratio for each distribution

            for i in range(3):
                dist[0][i], dist[1][i] = i, (weight[x][y][i] / math.sqrt(cov[x][y][i]))

            # using the distance we sort the 'dist' array - (Bubble Sort)
            for i in range(3):
                for j in range(0, 2 - i):
                    if dist[1][j] < dist[1][j + 1]:
                       dist[0][j + 1], dist[0][j] = dist[0][j], dist[0][j + 1]
                       dist[1][j + 1], dist[1][j] = dist[1][j], dist[1][j + 1]

            # Index of most probable and least probable distributions after above sorting
            # This is the index in the mean1, mean2, mean3, weight and cov array of the distributions

            most_prob = dist[0][0]
            least_prob = dist[0][2]

            # ADDING NEW DISTRIBUTION WHEN NO MATCH FOUND

            # If no match found then,
            # To add new distribution by replacing the least Probable Distribution
            if match == -1:
                mean1[x][y][least_prob] = frame[x][y][0]
                mean2[x][y][least_prob] = frame[x][y][1]
                mean3[x][y][least_prob] = frame[x][y][2]

                cov[x][y][least_prob] = 10000
                weight[x][y][least_prob] = 0.2 * weight[x][y][least_prob]
                weight[x][y][dist[0][:-1]] += 0.4 * weight[x][y][least_prob]
          

            # Pixel Classification as background or Foreground
            # B is our Background Threshold distribution

            B = 0
            sum_threshold = 0
            for i in range(3):
                sum_threshold += weight[x][y][dist[0][i]]
                B += 1
                if sum_threshold > T:
                    break

            # We will classify the Pixel as foreground in two cases :
            # Case-1- No match found for current Pixel value,
            # Case-2- Match found for current Pixel value But it is outside threshold background region B.

            # We will classify the Pixel as background in following case :
            # Case-3- Match found for current Pixel value And it is within the threshold background region B.

            if match == -1:
                fg[x][y][:] = frame[x][y][:]
            else:
                for i in range(B):
                    if match == dist[0][i]:
                        bg[x][y][:] = frame[x][y][:]
                        break
                    else:
                        fg[x][y][:] = frame[x][y][:]


    # cv2.imshow('fg',fg)
    cv2.imwrite(r"bg_frame%d.jpg" % count, bg);
    med = cv2.medianBlur(fg, 3)
    cv2.imwrite(r"fg_frame%d.jpg" % count, med);

    count += 1
    if cv2.waitKey(1000) == 27 & 0xFF:
        break

count = 0
while count <= 997:
    img1 = cv2.imread(r"C:\Users\Ravi\Desktop\fg8.2\frame%d.jpg" % count, 0)
    video1.write(img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

count = 0
while count <= 997:
    img1 = cv2.imread(r"C:\Users\Ravi\Desktop\bg8.2\frame%d.jpg" % count, 0)
    video2.write(img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

capture.release()
cv2.destroyAllWindows()
video1.release()
video2.release()
