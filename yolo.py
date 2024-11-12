import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Tải mô hình YOLOv8
model = YOLO('yolov8s.pt')  # Tải mô hình YOLOv8 nhỏ

# Định nghĩa màu cho mỗi lớp
colors = {
    0: (255, 255, 0),    # Màu đỏ cho lớp 0 (ví dụ: người)
    1: (255, 0, 0),    # Màu xanh lá cho lớp 1 (ví dụ: xe đạp)
    2: (0, 0, 255),    # Màu xanh dương cho lớp 2 (ví dụ: ô tô)
    # Thêm màu cho các lớp khác nếu cần
}

# Mở camera
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định
if not cap.isOpened():
    print("Lỗi: Không thể mở video.")
    exit()

cap.set(3, 640)  # Đặt chiều rộng
cap.set(4, 480)  # Đặt chiều cao

while True:
    ret, image = cap.read()  # Đọc khung hình từ camera

    if not ret:  # Kiểm tra xem khung hình đã được chụp thành công chưa
        break

   #Lật ngược hình ảnh theo chiều dọc
    image = cv2.flip(image, 2)  # 0 có nghĩa là lật quanh trục x (lộn ngược)

    # Phát hiện đối tượng
    results = model.predict(image)

    annotator = Annotator(image)  # Di chuyển ra ngoài vòng lặp để tăng hiệu suất
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Lấy tọa độ hộp giới hạn
            b = box.xyxy[0].numpy()  # x1, y1, x2, y2
            c = box.cls  # ID lớp
            color = colors.get(int(c), (255, 0, 0))  # Mặc định là trắng nếu lớp không có màu

            # Vẽ hộp giới hạn có màu lên hình ảnh
            annotator.box_label(b, model.names[int(c)], color=color)

    # Hiển thị hình ảnh có chú thích
    img = annotator.result()
    img_rez = cv2.resize(img, (640, 480))
    cv2.imshow("YOLOv8 Phát hiện Đối tượng với Hộp Màu", img_rez)

    # Thoát vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
