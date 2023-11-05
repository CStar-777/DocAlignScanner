from DocAlignScanner.config import *
from DocAlignScanner.simple_scan import *


class DocAlignScanner(QWidget):
    def __init__(self):
        super(DocAlignScanner, self).__init__()
        self.img = None
        self.scaled_img = None
        self.point = QPoint(0, 0)
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.scale = 0.5
        self.width = None
        self.height = None
        self.depth = None
        self.rate = None
        # self.Labelimg = QLabel(self)

    def init_ui(self):
        self.setWindowTitle("DocAlignScanner")

    def set_image(self, img_path):
        self.img = QPixmap(img_path)
        self.scaled_img = self.img.scaled(self.size())

        self.height = self.img.height()
        self.width = self.img.width()
        self.rate = self.width / self.height
    # 重绘事件定义
    def paintEvent(self, e):
        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale*self.rate, self.scale)
            painter.drawPixmap(self.point, self.scaled_img)
            painter.end()
    # 鼠标点击、移动操作
    def mouseMoveEvent(self, e):
        if self.left_click:
            self.end_pos = e.pos() - self.start_pos
            self.point = self.point + self.end_pos
            self.start_pos = e.pos()
            self.repaint()
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False
    # 鼠标滚轮缩放
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        self.old_scale = self.scale
        self.x, self.y = event.x(), event.y()
        self.wheel_flag = True
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.08
        else:  # 滚轮下滚
            self.scale *= 0.92
        if self.scale < 0.3:
            self.scale = 0.3
        self.adjustSize()
        self.update()


class MainDemo(QWidget):
    def __init__(self):
        super(MainDemo, self).__init__()
        self.img_name = None
        self.orig_img = None
        self.en_img = None

        self.setWindowTitle("星星文档扫描")
        # self.setFixedSize(1000, 600)
        self.setGeometry(300, 200, 900, 700)
        self.setWindowIcon(QIcon("./icons/star.svg"))

        self.open_file = QPushButton("Open")
        self.open_file.setToolTip("导入图像")
        self.open_file.clicked.connect(self.open_image)
        self.open_file.setFixedSize(70, 30)

        self.save_file = QPushButton("Save")
        self.save_file.setToolTip("保存图像")
        self.save_file.clicked.connect(self.save_image)
        self.save_file.setFixedSize(70, 30)

        self.zoom_in = QPushButton("")
        self.zoom_in.clicked.connect(self.large_click)
        self.zoom_in.setFixedSize(30, 30)
        in_icon = QIcon("./icons/zoom_in.svg")
        self.zoom_in.setIcon(in_icon)
        self.zoom_in.setIconSize(QSize(30, 30))

        self.zoom_out = QPushButton("")
        self.zoom_out.clicked.connect(self.small_click)
        self.zoom_out.setFixedSize(30, 30)
        out_icon = QIcon("./icons/zoom_out.svg")
        self.zoom_out.setIcon(out_icon)
        self.zoom_out.setIconSize(QSize(30, 30))

        # 手工调整角点
        self.btn1 = QPushButton(self)
        self.btn1.setText("手动调整")
        self.btn1.clicked.connect(self.manual_adjust)
        self.btn1.setFixedSize(70, 30)

        # 旋转
        self.btn2 = QPushButton(self)
        self.btn2.setText("旋转")
        self.btn2.clicked.connect(self.rotate_img)
        self.btn2.setFixedSize(70, 30)

        # 图像增强
        self.btn3 = QPushButton(self)
        self.btn3.setText("图像增强")
        self.btn3.clicked.connect(self.enhance_img)
        self.btn3.setFixedSize(70, 30)

        # 文档扫描对齐
        self.btn4 = QPushButton(self)
        self.btn4.setText("对齐扫描(r:旋转/e：图像增强/s:保存/esc:退出)")
        self.btn4.clicked.connect(self.scan_img)
        self.btn4.setFixedSize(280, 30)



        # 控件布局
        # 水平布局
        w = QWidget(self)
        layout = QHBoxLayout()
        layout.addWidget(self.open_file)
        layout.addWidget(self.save_file)
        layout.addWidget(self.zoom_in)
        layout.addWidget(self.zoom_out)
        layout.setAlignment(Qt.AlignLeft)
        w.setLayout(layout)
        w.setFixedSize(550, 50)

        w2 = QWidget(self)
        Hbox = QHBoxLayout()
        Hbox.addWidget(self.btn1)
        Hbox.addWidget(self.btn2)
        Hbox.addWidget(self.btn3)
        Hbox.addWidget(self.btn4)
        Hbox.setAlignment(Qt.AlignBottom)
        w2.setLayout(Hbox)
        w2.setFixedSize(550, 40)

        self.box = DocAlignScanner()
        # self.box.resize(500, 600)
        # 垂直布局
        Vbox = QVBoxLayout()
        Vbox.addWidget(w)
        Vbox.addWidget(self.box)
        Vbox.addWidget(w2)
        self.setLayout(Vbox)

    # 函数定义
    # 导入图片
    def open_image(self):
        img_name, _ = QFileDialog.getOpenFileName(self, "导入图像", "*.jpg;;*.png;;*.jpeg")
        self.img_name = img_name
        self.box.set_image(img_name)
        self.orig_img = self.box.scaled_img.copy()
    # 保存图片
    def save_image(self):
        img_name, _ = QFileDialog.getSaveFileName(self, "保存图像", "*.jpg;;*.png;;*.jpeg")
        self.box.set_image(img_name)
    # 点击放大
    def large_click(self):
        if self.box.scale < 2:
            self.box.scale += 0.1
            self.box.adjustSize()
            self.update()
    # 点击缩小
    def small_click(self):
        if self.box.scale > 0.1:
            self.box.scale -= 0.2
            self.box.adjustSize()
            self.update()

    # 手动调整角点
    def manual_adjust(self):
        msg_box = QMessageBox(QMessageBox.Information, '提示', '该功能等待后续实现')
        msg_box.exec_()

    # 旋转图像
    def rotate_img(self):
        transform = QTransform()
        transform.rotate(90)
        self.box.scaled_img = self.box.scaled_img.transformed(transform)
        self.orig_img = self.box.scaled_img
        self.update()

    # 图像增强
    def enhance_img(self):
        qimg = self.orig_img.toImage()
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        cvimg = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        cvimg = cvimg[..., :3]
        # cvimg = cv2.imread(self.img_name, cv2.IMREAD_COLOR)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.adaptiveThreshold(cvimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        cv2.namedWindow('enhance', cv2.WINDOW_NORMAL)
        cv2.imshow('enhance',gray_img)

    # 文档对齐扫描
    def scan_img(self):
        def enhance(img):
            dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 自适应阈值
            th = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
            return th

        def cv_show(img):
            cv2.imshow('scanner', img)
            k = cv2.waitKey(0)
            if k == 27:  # 按ESC退出
                cv2.destroyAllWindows()
            elif k == ord('r'):  # 按r旋转结果
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv_show(rotated)
            elif k == ord('e'):  # 按e显示增强图像
                c = len(img.shape)
                if c == 3:
                    img = enhance(img)
                    cv_show(img)
                else:
                    img = scanned_img
                    cv_show(img)
            elif k == ord('s'):  # 按s保存图片
                file_name = os.path.basename(self.img_name)
                file_name, file_ext = os.path.splitext(file_name)
                save_path = "outputs//"+file_name+"_ss.jpg"
                cv2.imwrite(save_path, img)

        msg_box = QMessageBox(QMessageBox.Information, '使用方法',
                              '键盘控制(英文输入)：\n按"s"：保存\n按"r"：旋转\n按"e"：图像增强\n按"esc"：关闭')
        msg_box.exec_()

        cvimg = cv2.imread(self.img_name)
        scanned_img = scan(cvimg)
        cv2.namedWindow('scanner', cv2.WINDOW_NORMAL)
        cv_show(scanned_img)


    # opencv图像转为qimage（未使用，效果不好）
    def Opencv2QImage(cvimg):
        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
        return cvimg

    # Qpixmap转opencv（未使用，效果不好）
    def QPixmap2Opencv(pixmap):
        qimg = pixmap.toImage()
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]
        return result