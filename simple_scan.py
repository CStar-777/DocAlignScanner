from DocAlignScanner.config import *
'''
用于生成文档对齐扫描结果的py文件
'''

# 按顺序重新排列坐标（左上、右上、右下、左下）
def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # 左上角的点的和最小
    rect[0] = pts[np.argmin(s)]
    # 右下角的点的和最大
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # 右上角的差值最小
    rect[1] = pts[np.argmin(diff)]
    # 左下角的差值最大
    rect[3] = pts[np.argmax(diff)]
    # 返回有序坐标
    return rect.astype('int').tolist()


def scan(img):
    # 将图像调整为适合运行的大小
    # 不是说一定要调整，但是按原图大小处理的速度会很慢，
    # 相应的，调整大小后生成的结果图像质量没有使用原图的好
    # dim_limit = 1080
    dim_limit = 4000
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    # 复制图像副本，方便后续使用
    orig_img = img.copy()
    # 重复闭运算以从文档中删除文本
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # 使用grabCut区分前景背景
    # 未使用，使用速度会慢（质量会好一点？）
    # mask = np.zeros(img.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask2[:, :, np.newaxis]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (11, 11), 0)


    # 边缘检测
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    # 查找检测到的边缘的轮廓
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 只保留检测到的最大轮廓
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # 通过轮廓近似检测边缘
    if len(page) == 0:
        return orig_img
    # 在轮廓上循环
    for c in page:
        # 近似轮廓
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # 如果找到4个近似角点
        if len(corners) == 4:
            break
    # 对角点进行排序并将其转换为所需形状
    corners = sorted(np.concatenate(corners).tolist())
    # 对检测的4个角点，重新排列角点的顺序
    corners = order_points(corners)

    # 查找目标的坐标
    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    # 查找最大宽度
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    # 查找最大高度
    h = max(int(h1), int(h2))

    # 最终的目标坐标
    destination_corners = order_points(np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]))
    h, w = orig_img.shape[:2]

    # 图像透视变换
    '''
    方法一:getPerspectiveTransform
    基于 findHomography的，当你只有四个点时，并且你知道它们是正确的点，getPerspectiveTransform 非常有用
    '''
    # 根据源图像和目标图像上的四对点坐标来计算从原图像透视变换到目标头像的透视变换矩阵
    # homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # 对输入图像进行透视变换
    # final = cv2.warpPerspective(orig_img, np.float32(homography), (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    '''
    方法二：findHomography
    找出两组特征点之间最好的变换关系，它用了比最小均方差更好的一种方法，RANSAC，
    这种方法能够排除一些离群点，如果你的数据50以上，RANSAC将能够建立一个可靠的变换关系
    '''
    # 找到两个平面之间的转换矩阵
    homography, mask = cv2.findHomography(np.float32(corners), np.float32(destination_corners), method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
    # 对输入图像进行透视变换
    un_warped = cv2.warpPerspective(orig_img, np.float32(homography), (w, h), flags=cv2.INTER_LINEAR)
    # 剪裁
    final = un_warped[:destination_corners[2][1], :destination_corners[2][0]]
    return final



