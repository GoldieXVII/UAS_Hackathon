from picamera import PiCamera
from time import sleep
import os
from datetime import datetime

camera = PiCamera()

savePath = '/home/pi/images/'

if not os.path.exists(savePath):
    os.makedirs(savePath)

def take_photo():
    timestamp =  datetime.now().strftime("%Y%m%d_%H%M%S")
    filePath = os.path.join(savePath, f"photo_{timestamp}.jpg")
    camera.capture(filePath)
    print(f"Photo saved at {filePath}")

for i in range(5):
    take_photo()
    sleep(10)
