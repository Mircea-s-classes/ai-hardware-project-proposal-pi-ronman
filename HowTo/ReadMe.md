<b>Bird Detection for the Pi5 with AI hat is a python file called camera_hailo_yolo_birds.py</b>
<br>This Python script is designed to work with a usb camera as the input device

To run the file, enter the code below in the Pi terminal window <br>
<br>cd ~/hailo-rpi5-examples
<br>source venv_hailo_rpi_examples/bin/activate
<br>python camera_hailo_yolo_birds.py

To view the live stream, open a internet browser window and enter the local steraming address below

http://192.168.1.163:8080/


<b>To run Bird Detection on Pi5 without the AI hat
<br>This Python script is designed to work with a usb camera as the input device

<br>To run the file, enter the code below in the Pi terminal window
<br>source ~/yolo-venv/bin/activate
<br>pip install psutil matplotlib

<br>python Camera_NOHailo_yolo_bird.py --source 0 --model yolo11n.pt --logfps ~/fps_cpu_log.csv

