import cv2
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM
from bresenham import bresenham
from scipy.interpolate import interp1d

def divide_arc_length(X, Y, n):
    """
    Divides the arc length of the points (X, Y) into n equal segments.

    Parameters:
    X (array): The x-coordinates of the points.
    Y (array): The y-coordinates of the points.
    n (int): The number of equal segments.

    Returns:
    list: A list of x coordinates that divide the arc into n equal segments.
    """
    # Calculate differences between consecutive points
    dx = np.diff(X)
    dy = np.diff(Y)
    
    # Calculate the arc length increments
    ds = np.sqrt(dx**2 + dy**2)
    
    # Cumulative arc length
    s = np.concatenate(([0], np.cumsum(ds)))

    # Total arc length
    L = s[-1]
    Delta_L = L / n

    # Find the points x_i
    x_points = [X[0]]  # Start with the initial point

    for i in range(1, n):
        target_length = i * Delta_L
        
        # Interpolating the x value at the target arc length
        idx = np.searchsorted(s, target_length)
        x0, x1 = X[idx-1], X[idx]
        s0, s1 = s[idx-1], s[idx]
        
        # Linear interpolation for the x value at the target arc length
        x_interp = x0 + (target_length - s0) * (x1 - x0) / (s1 - s0)
        x_points.append(x_interp)
    
    x_points.append(X[-1])  # Include the endpoint

    return x_points

def reshape_array_with_interpolation(original_array, new_size, kind='linear'):
    """
    Reshape an array to a new size using interpolation.

    Parameters:
    - original_array: The original numpy array.
    - new_size: The desired size of the new array.
    - kind: The type of interpolation (e.g., 'linear', 'cubic').

    Returns:
    - A new numpy array of shape (new_size,).
    """

    # Original indices based on the original array size
    original_indices = np.linspace(0, len(original_array) - 1, len(original_array))

    # New indices for the desired output shape
    new_indices = np.linspace(0, len(original_array) - 1, new_size)

    # Use interpolation
    interpolation_function = interp1d(original_indices, original_array, kind=kind)

    # Interpolate to find new values
    new_array = interpolation_function(new_indices)

    return np.round(new_array)

def pad_binary_image_with_ones(image):
    """
    Pad a binary image with 1's on all sides, doubling its size.

    Parameters:
    - image: a 2D numpy array representing the binary image.

    Returns:
    - A new 2D numpy array representing the padded image.
    """
    # Get the original image dimensions
    original_height, original_width = image.shape
    
    # Create a new array of ones with double the dimensions of the original image
    new_height = 2 * original_height
    new_width = 2 * original_width
    padded_image = np.ones((new_height, new_width), dtype=image.dtype) + 254
    
    # Copy the original image into the center of the new array
    start_row = original_height // 2
    start_col = original_width // 2
    padded_image[start_row:start_row + original_height, start_col:start_col + original_width] = image
    
    return padded_image

def find_distance_d(X, y, X_new, y_hat, step):
    # Starting point for the distance d
    d = 0
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    found = False

    # Increment d until all points are covered or max_iterations is reached
    while iteration < max_iterations and not found:
        # Create two functions shifted by d
        upper_function = y_hat + d
        lower_function = y_hat - d
        
        # Check if all y points are within the bounds
        all_points_covered = np.all([(y[i] <= upper_function[np.argmin(np.abs(X_new - X[i]))]) and 
                                    (y[i] >= lower_function[np.argmin(np.abs(X_new - X[i]))]) for i in range(len(X_new))])
            
        if all_points_covered:
            found = True
        else:
            d += step  # Increment d
            iteration += 1

    return int(np.ceil(2*d))

def calculate_derivative(y_values):
    dy = np.zeros(y_values.shape)
    dy[0] = y_values[1] - y_values[0]  # Forward difference
    dy[-1] = y_values[-1] - y_values[-2]  # Backward difference
    dy[1:-1] = (y_values[2:] - y_values[:-2]) / 2  # Central difference
    return dy

def find_perpendicular_points(y_values, x_values, d):
    dy = calculate_derivative(y_values)
    perpendicular_points = []
    
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        slope = dy[i]
        if slope != 0:
            perp_slope = -1 / slope
        else:
            perp_slope = np.inf
        
        if np.isinf(perp_slope):  # Vertical line
            points = [(round(x), round(y - d)), (round(x), round(y + d))]
        else:
            # y = mx + c form for perpendicular line
            c = y - perp_slope * x
            # Solve for points that are distance d away from (x, y)
            delta = d / np.sqrt(1 + perp_slope**2)
            x1, x2 = x + delta, x - delta
            y1, y2 = perp_slope * x1 + c, perp_slope * x2 + c
            points = [(round(x1), round(y1)), (round(x2), round(y2))]
        
        perpendicular_points.append(points)
    
    return perpendicular_points

def uncurve_text_tight(input_path, output_path, n_splines, arc_equal=False):
    # Load image, grayscale it, Otsu's threshold
    image = cv2.imread(input_path)
    print(1)
    print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(2)
    print(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print(3)
    print(thresh)
    #thresh = pad_binary_image_with_ones(thresh)
    print(4)
    print(thresh)
    
    # Dilation & Erosion to fill holes inside the letters
    kernel = np.ones((3, 3), np.uint8)
    #thresh = cv2.erode(thresh, kernel, iterations=1)
    print(5)
    #print(thresh)
    #thresh = cv2.dilate(thresh, kernel, iterations=1)
    print(6)
    #print(thresh)
    
    black_pixels = np.column_stack(np.where(thresh == 0))

    print(7)
    print(np.where(thresh == 0))
    print(black_pixels)
    leftmost_x = np.min(black_pixels[:, 1]) - int(0.05*(np.max(black_pixels[:, 1]) - np.min(black_pixels[:, 1])))
    print(leftmost_x)
    rightmost_x = np.max(black_pixels[:, 1]) + int(0.05*(np.max(black_pixels[:, 1]) - np.min(black_pixels[:, 1])))
    print(rightmost_x)
    X = black_pixels[:, 1].reshape(-1, 1)
    print(X)
    y = black_pixels[:, 0]
    print(y)
    
    # gam = LinearGAM(n_splines = n_splines)
    # print(gam)
    # gam.fit(X, y)
    # print(gam)
    #
    # if arc_equal!=True:
    #     X_new = np.linspace(leftmost_x, rightmost_x, num = rightmost_x - leftmost_x)
    # else:
    #     # Generate a dense set of points for accurate arc length calculation
    #     X_dense = np.linspace(leftmost_x, rightmost_x, num = rightmost_x - leftmost_x)
    #     Y_dense = gam.predict(X_dense)
    #
    #     # Interval and number of segments
    #     n = rightmost_x - leftmost_x  # Number of equal segments
    #
    #     # Get the points dividing the arc length into equal segments
    #     X_new = divide_arc_length(X_dense, Y_dense, n)
    #
    # # Create the offset necessary to un-curve the text
    # y_hat = gam.predict(X_new)
    X_new = [340.3015921211977, 339.4525458323973, 338.5695500184713, 337.65287364849837, 336.7027959509809, 335.7196063287891, 334.70360427100627, 333.65509926170154, 332.5744106856577, 331.461867731084, 330.31780928934177, 329.1425838517151, 327.936549403257, 326.7000733137436, 325.43353222576974, 324.1373119400201, 322.8118072977504, 321.45742206051534, 320.0745687871782, 318.6636687082421, 317.2251515975388, 315.7594556413154, 314.2670273047588, 312.7483211959977, 311.2037999276248, 309.6339339757807, 308.0392015368419, 306.4200883817578, 304.77708770808016, 303.1106999897303, 301.4214328245499, 299.7098007796817, 297.97632523482764, 296.2215342234311, 294.4459622718333, 292.650150236451, 290.83464513902663, 289.0, 287.1467736700531, 285.27553065987865, 283.3868409682247, 281.4812799082674, 279.55942793236534, 277.62187045524763, 275.6691976756914, 273.70200439674124, 271.72088984452665, 269.7264574857316, 267.71931484377313, 265.7000733137436, 263.66934797617387, 261.6277574096739, 259.5759235025075, 257.51447126315946, 255.44402862995153, 253.36522627976666, 251.27869743593848, 249.1850776753655, 247.08500473490818, 244.97911831712798, 242.868059895428, 240.75247251865372, 238.63300061521429, 236.51028979678327, 234.38498666163918, 232.25773859770513, 230.1291935853486, 228.0, 225.8708064146514, 223.74226140229487, 221.61501333836085, 219.4897102032167, 217.3669993847857, 215.24752748134625, 213.131940104572, 211.02088168287202, 208.91499526509185, 206.8149223246345, 204.72130256406155, 202.63477372023337, 200.55597137004847, 198.48552873684054, 196.42407649749245, 194.37224259032612, 192.33065202382613, 190.2999266862564, 188.2806851562269, 186.2735425142684, 184.27911015547338, 182.29799560325873, 180.3308023243086, 178.37812954475237, 176.4405720676347, 174.51872009173255, 172.6131590317753, 170.72446934012132, 168.8532263299469, 167.00000000000003, 165.16535486097337, 163.349849763549, 161.5540377281667, 159.7784657765689, 158.02367476517236, 156.2901992203183, 154.5785671754501, 152.8893000102697, 151.22291229191984, 149.5799116182422, 147.96079846315814, 146.3660660242193, 144.7962000723752, 143.25167880400232, 141.73297269524122, 140.24054435868453, 138.7748484024612, 137.33633129175792, 135.92543121282182, 134.54257793948472, 133.18819270224955, 131.86268805997992, 130.56646777423026, 129.2999266862564, 128.063450596743, 126.85741614828493, 125.68219071065828, 124.53813226891603, 123.4255893143423, 122.34490073829848, 121.29639572899372, 120.2803936712109, 119.29720404901913, 118.34712635150161, 117.43044998152871, 116.54745416760271, 115.69840787880227]
    y_hat = np.array([276.6691976756914, 278.62187045524763, 280.55942793236534, 282.4812799082674, 284.3868409682247, 286.27553065987865, 288.1467736700531, 290.0, 291.83464513902663, 293.650150236451, 295.4459622718333, 297.22153422343115, 298.97632523482764, 300.7098007796817, 302.42143282454987, 304.1106999897303, 305.77708770808016, 307.4200883817578, 309.0392015368419, 310.6339339757807, 312.2037999276248, 313.7483211959977, 315.2670273047588, 316.7594556413154, 318.2251515975388, 319.6636687082421, 321.0745687871782, 322.45742206051534, 323.8118072977504, 325.1373119400201, 326.43353222576974, 327.7000733137436, 328.936549403257, 330.1425838517151, 331.31780928934177, 332.461867731084, 333.5744106856577, 334.6550992617015, 335.70360427100627, 336.7196063287891, 337.7027959509809, 338.65287364849837, 339.5695500184713, 340.4525458323973, 341.3015921211977, 342.11643025714807, 342.8968120326586, 343.64249973588085, 344.35326622311663, 345.02889498800874, 345.6691802274903, 346.27392690447493, 346.84295080726633, 347.3760786056716, 347.87314790379867, 348.33400728952427, 348.758516380615, 349.1465458674894, 349.4979775526068, 349.8127043864716, 350.0906305002413, 350.33167123492933, 350.535753167193, 350.7028141316986, 350.832803240058, 350.9256808963297, 350.98141880907974, 351.0, 350.98141880907974, 350.9256808963297, 350.832803240058, 350.7028141316986, 350.535753167193, 350.33167123492933, 350.0906305002413, 349.8127043864716, 349.4979775526068, 349.1465458674894, 348.758516380615, 348.33400728952427, 347.87314790379867, 347.3760786056716, 346.84295080726633, 346.27392690447493, 345.6691802274903, 345.02889498800874, 344.35326622311663, 343.64249973588085, 342.8968120326586, 342.11643025714807, 341.3015921211977, 340.4525458323973, 339.5695500184713, 338.65287364849837, 337.7027959509809, 336.7196063287891, 335.70360427100627, 334.65509926170154, 333.5744106856577, 332.461867731084, 331.31780928934177, 330.1425838517151, 328.936549403257, 327.7000733137436, 326.4335322257697, 325.1373119400201, 323.8118072977505, 322.45742206051534, 321.0745687871782, 319.6636687082421, 318.2251515975388, 316.7594556413154, 315.2670273047588, 313.7483211959976, 312.2037999276248, 310.63393397578074, 309.0392015368419, 307.4200883817578, 305.77708770808016, 304.11069998973034, 302.42143282454987, 300.70980077968176, 298.9763252348276, 297.22153422343115, 295.4459622718333, 293.650150236451, 291.83464513902663, 290.0, 288.1467736700531, 286.27553065987865, 284.3868409682247, 282.4812799082674, 280.55942793236534, 278.62187045524763, 276.6691976756914])

    # Plot the image with text curve overlay
    plt.imshow(thresh, cmap='gray')
    plt.plot(X_new, y_hat, color='red')
    plt.axis('off')
    plt.subplots_adjust(bottom = 0, left = 0, right = 1, top = 1)
    plt.show()
    
    # Calculate height of text
    d = find_distance_d(X, y, X_new, y_hat, step = 0.5)
    
    # Create an image full of zeros
    dewarp_image = np.zeros(((2*d+1), len(X_new)), dtype=np.uint8) + 255
    
    # Calculate perpendicular points
    perpendicular_points = find_perpendicular_points(y_hat, X_new, d)
    my_iter = 0

    for points in perpendicular_points:
        x1, y1, x2, y2 = [element for tup in points for element in tup]
        if y1 > y2:  # If y1 is below y2, swap them to ensure top-to-bottom interpolation
            y1, y2 = y2, y1
            x1, x2 = x2, x1
        # Extract pixel values
        bresenham_list = list(bresenham(x1, y1, x2, y2))
        # Extract pixel values, ensuring they are within the bounds of the image
        pixel_values = []
        for x, y in bresenham_list:
            pixel_values.append(thresh[y, x])
        dewarp_image[:, my_iter] = reshape_array_with_interpolation(np.array(pixel_values), (2*d+1), kind='linear')
        my_iter += 1
  
    # Plot the original image
    plt.imshow(thresh, cmap='gray', extent=[0, thresh.shape[1], thresh.shape[0], 0])
    
    # Plot the y_hat line
    plt.plot(X_new, y_hat, color='red')
    
    # Plot perpendicular points
    for points in perpendicular_points:
      plt.plot([x[0] for x in points], [x[1] for x in points], color='blue', alpha=0.5)
    
    plt.axis('off')
    plt.subplots_adjust(bottom = 0, left = 0, right = 1, top = 1)
    plt.show()
    
    # Plot the final image
    plt.imshow(dewarp_image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplots_adjust(bottom = 0, left = 0, right = 1, top = 1)
    plt.show()
    
    # Save image to desired directory
    cv2.imwrite(output_path, dewarp_image)

def uncurve_text(input_path, output_path, n_splines, arc_equal=False):
    # Load image, grayscale it, Otsu's threshold
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Dilation & Erosion to fill holes inside the letters
    kernel = np.ones((3, 3), np.uint8)
    #thresh = cv2.erode(thresh, kernel, iterations=1)
    #thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    black_pixels = np.column_stack(np.where(thresh == 0))
    leftmost_x, rightmost_x = np.min(black_pixels[:, 1]), np.max(black_pixels[:, 1])
    X = black_pixels[:, 1].reshape(-1, 1)
    y = thresh.shape[0] - black_pixels[:, 0]
    
    gam = LinearGAM(n_splines = n_splines)
    gam.fit(X, y)
    
    # Create the offset necessary to un-curve the text
    if arc_equal!=True:
        X_new = np.linspace(leftmost_x, rightmost_x, num = rightmost_x - leftmost_x + 1)
    else:
        # Generate a dense set of points for accurate arc length calculation
        X_dense = np.linspace(leftmost_x, rightmost_x, num = rightmost_x - leftmost_x + 1)
        Y_dense = gam.predict(X_dense)

        # Interval and number of segments
        n = rightmost_x - leftmost_x + 1 # Number of equal segments

        # Get the points dividing the arc length into equal segments
        X_new = divide_arc_length(X_dense, Y_dense, n)
    
    # Create the offset necessary to un-curve the text
    y_hat = gam.predict(X_new)
    
    # Plot the image with text curve overlay
    plt.imshow(image[:,:,::-1])
    plt.plot(X_new, (thresh.shape[0] - y_hat), color='red')
    plt.axis('off')
    plt.subplots_adjust(bottom = 0, left = 0, right = 1, top = 1)
    plt.show()

    # Roll each column to align the text
    for i in range(leftmost_x, rightmost_x + 1):
        image[:, i, 0] = np.roll(image[:, i, 0], round(y_hat[i - leftmost_x] - thresh.shape[0]/2))
        image[:, i, 1] = np.roll(image[:, i, 1], round(y_hat[i - leftmost_x] - thresh.shape[0]/2))
        image[:, i, 2] = np.roll(image[:, i, 2], round(y_hat[i - leftmost_x] - thresh.shape[0]/2))
  
    # Plot the final image
    plt.imshow(image[:,:,::-1])
    plt.axis('off')
    plt.subplots_adjust(bottom = 0, left = 0, right = 1, top = 1)
    plt.show()
    
    # Save image to desired directory
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    
    input_path = 'images/stamp3.jpg'
    output_path = 'images/100.png' #'sports_output.png'
    final_path = 'sports_final.png'
    n1_splines = 6
    n2_splines = 9
    uncurve_text_tight(input_path, output_path, n1_splines, arc_equal=True)
    #uncurve_text(output_path, final_path, n2_splines, arc_equal=False)

#python dewarp-rectify.py ./sample.png ./output.png