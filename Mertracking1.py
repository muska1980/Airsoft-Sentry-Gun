import cv2
import time
import threading
import sys
import termios
import contextlib
import RPi.GPIO as GPIO
import imutils
import board

from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

# Konfig
MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False
MAX_STEPS_X = 20
MAX_STEPS_Y = 10
RELAY_PIN = 22


@contextlib.contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

class VideoUtils(object):
    @staticmethod
    def live_video(camera_port=0):
        """
        Zeigt ein Livebild von der Kamera an.
        """
        video_capture = cv2.VideoCapture(camera_port)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            cv2.imshow('Live Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def find_motion(callback, camera_port=0, show_video=False):
        """
        Erkennt Bewegungen per Differenzanalyse und ruft callback mit Kontur und Bild auf.
        """
        camera = cv2.VideoCapture(camera_port)
        time.sleep(0.5)
        firstFrame = None

        while True:
            grabbed, frame = camera.read()
            if not grabbed:
                break

            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if firstFrame is None:
                firstFrame = gray
                continue

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            c = VideoUtils.get_best_contour(thresh, threshold=5000)

            if c is not None:
                callback(c, frame)

            if show_video:
                cv2.imshow("Motion", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        camera.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_best_contour(imgmask, threshold):
        """
        Gibt den größten Bereich über Schwelle zurück oder None.
        """
        contours = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt
    
class Mertracking(object):
    def __init__(self, friendly_mode=True):
        self.friendly_mode = friendly_mode
        self.kit = MotorKit(i2c=board.I2C())
        self.sm_x = self.kit.stepper1
        self.sm_y = self.kit.stepper2
        self.current_x_steps = 0
        self.current_y_steps = 0

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    def interactive(self):
        print("Controls: (a) ←, (d) →, (w) ↑, (s) ↓, (q) quit, (Enter) fire")
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break
                    if ch == "a":
                        self._move(self.sm_x, 5, not MOTOR_X_REVERSED)
                    elif ch == "d":
                        self._move(self.sm_x, 5, MOTOR_X_REVERSED)
                    elif ch == "w":
                        self._move(self.sm_y, 5, not MOTOR_Y_REVERSED)
                    elif ch == "s":
                        self._move(self.sm_y, 5, MOTOR_Y_REVERSED)
                    elif ch == "\n":
                        self.fire()
            except (KeyboardInterrupt, EOFError):
                print("Abgebrochen...")
        self.release()
    
    @staticmethod
    def fire():
        """
        Aktiviert das Relais zum Feuern.
        """
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    @staticmethod
    def move_forward(motor, steps):
        """
        Bewegt den Schrittmotor vorwärts.
        """
        for _ in range(steps):
            motor.onestep(direction=stepper.FORWARD, style=stepper.MICROSTEP)
            time.sleep(0.01)

    @staticmethod
    def move_backward(motor, steps):
        """
        Bewegt den Schrittmotor rückwärts.
        """
        for _ in range(steps):
            motor.onestep(direction=stepper.BACKWARD, style=stepper.MICROSTEP)
            time.sleep(0.01)

    def _move(self, motor, steps=1, forward=True):
        direction = stepper.FORWARD if forward else stepper.BACKWARD
        for _ in range(steps):
            motor.onestep(direction=direction, style=stepper.MICROSTEP)
            time.sleep(0.01)

    def release(self):
        self.kit.stepper1.release()
        self.kit.stepper2.release()

    def motion_detection(self, show_video=False):
        camera = cv2.VideoCapture(0)
        time.sleep(0.5)
        firstFrame = None

        while True:
            grabbed, frame = camera.read()
            if not grabbed:
                break

            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if firstFrame is None:
                firstFrame = gray
                continue

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for c in contours:
                if cv2.contourArea(c) > 5000:
                    (x, y, w, h) = cv2.boundingRect(c)
                    self._move_axis((x, y, w, h), frame)

            if show_video:
                cv2.imshow("Motion", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.release()
        cv2.destroyAllWindows()

    def _move_axis(self, rect, frame):
        x, y, w, h = rect
        v_h, v_w = frame.shape[:2]
        target_x = (2 * MAX_STEPS_X * (x + w / 2) / v_w) - MAX_STEPS_X
        target_y = (2 * MAX_STEPS_Y * (y + h / 2) / v_h) - MAX_STEPS_Y

        delta_x = int(target_x - self.current_x_steps)
        delta_y = int(target_y - self.current_y_steps)

        if delta_x != 0:
            self._move(self.sm_x, abs(delta_x), forward=(delta_x > 0) ^ MOTOR_X_REVERSED)
            self.current_x_steps += delta_x

        if delta_y != 0:
            self._move(self.sm_y, abs(delta_y), forward=(delta_y > 0) ^ MOTOR_Y_REVERSED)
            self.current_y_steps += delta_y

        if not self.friendly_mode:
            if abs(delta_x) <= 2 and abs(delta_y) <= 2:
                self.fire()


if __name__ == "__main__":
    tracker = Mertracking(friendly_mode=False)

    mode = input("Modus wählen: (1) Motion Detection, (2) Manuell\n")
    if mode == "1":
        show = input("Live Video anzeigen? (y/n): ").lower() == "y"
        tracker.motion_detection(show_video=show)
    elif mode == "2":
        tracker.interactive()
    else:
        print("Ungültige Auswahl.")
