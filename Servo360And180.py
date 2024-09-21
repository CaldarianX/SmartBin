import pigpio
import time

# Connect to pigpio daemon
pi = pigpio.pi()

# GPIO pin for the 180-degree servo
servo_180_gpio = "YOURPIN"
# GPIO pin for the 360-degree servo
servo_360_gpio = "YOURPIN"

# Set initial positions
pi.set_servo_pulsewidth(servo_180_gpio, 1500)  # Neutral position for 180-degree servo
pi.set_servo_pulsewidth(servo_360_gpio, 1500)  # Neutral position for 360-degree servo

try:
    while True:
        print("180 Servo")
        # Control 180-degree servo
        # Move to 0 degrees
        pi.set_servo_pulsewidth(servo_180_gpio, 500)
        time.sleep(1)
        
        # Move to 90 degrees
        pi.set_servo_pulsewidth(servo_180_gpio, 1500)
        time.sleep(1)
        
        # Move to 180 degrees
        pi.set_servo_pulsewidth(servo_180_gpio, 2500)
        time.sleep(1)

        print("360 Servo")
        # Control 360-degree servo
        # Rotate clockwise
        pi.set_servo_pulsewidth(servo_360_gpio, 1000)
        time.sleep(1)
        
        # Stop
        pi.set_servo_pulsewidth(servo_360_gpio, 1500)
        time.sleep(1)
        
        # Rotate counterclockwise
        pi.set_servo_pulsewidth(servo_360_gpio, 2000)
        time.sleep(1)
        print("-------------------End of round-------------------")
except KeyboardInterrupt:
    # Stop servos
    pi.set_servo_pulsewidth(servo_180_gpio, 0)
    pi.set_servo_pulsewidth(servo_360_gpio, 0)
    pi.stop()
