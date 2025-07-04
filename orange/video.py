import cv2 as cv

# Buka kamera (0 berarti kamera default)
kamera = cv.VideoCapture(0)

if not kamera.isOpened():
    print("Kamera tidak dapat diakses!")
    exit()

while True:
    ret, frame = kamera.read()
    if not ret:
        print("Gagal membaca frame!")
        break
    # Tampilkan frame di jendela
    cv.imshow('Webcam', frame)
    # Tekan 'q' untuk keluar
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# Lepaskan kamera dan tutup jendela
kamera.release()
cv.destroyAllWindows()
