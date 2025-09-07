import scservo_sdk as scs
import time
from config import (
    SERIAL_PORT, BAUD_RATE, ADDR_TORQUE_ENABLE, ADDR_PRESENT_POSITION, 
    ADDR_GOAL_POSITION, ADDR_GOAL_ACC, ADDR_GOAL_SPEED, ADDR_MIDPOINT_OFFSET
)

class ServoHandler:
    def __init__(self):
        self.portHandler = scs.PortHandler(SERIAL_PORT)
        self.packetHandler = scs.PacketHandler(0)
        self.is_port_open = False

    def connect(self):
        """Opens the serial port and sets the baud rate."""
        if not self.portHandler.openPort():
            raise IOError(f"Failed to open port {SERIAL_PORT}")
        if not self.portHandler.setBaudRate(BAUD_RATE):
            raise IOError(f"Failed to set baud rate to {BAUD_RATE}")
        self.is_port_open = True
        print(f"Successfully connected to servos on {SERIAL_PORT} at {BAUD_RATE} bps.")

    def disconnect(self):
        """Closes the serial port."""
        if self.is_port_open:
            self.portHandler.closePort()
            self.is_port_open = False
            print("Servo port closed.")

    def _execute_servo_command(self, servo_id, addr, value, size=1, write=True):
        """Generic function to execute a read or write command."""
        if not self.is_port_open:
            print("Port is not open. Cannot send command.")
            return (None, None, None) if not write else (None, None)

        # Use a variable to store the read value, default to None
        read_value = None

        if write:
            if size == 1:
                result, error = self.packetHandler.write1ByteTxRx(self.portHandler, servo_id, addr, value)
            elif size == 2:
                result, error = self.packetHandler.write2ByteTxRx(self.portHandler, servo_id, addr, value)
            elif size == 4:
                result, error = self.packetHandler.write4ByteTxRx(self.portHandler, servo_id, addr, value)
            else:
                print(f"Error: Invalid write size {size}")
                return None, None
        else: # read
            if size == 1:
                read_value, result, error = self.packetHandler.read1ByteTxRx(self.portHandler, servo_id, addr)
            elif size == 2:
                read_value, result, error = self.packetHandler.read2ByteTxRx(self.portHandler, servo_id, addr)
            elif size == 4:
                read_value, result, error = self.packetHandler.read4ByteTxRx(self.portHandler, servo_id, addr)
            else:
                print(f"Error: Invalid read size {size}")
                return None, None, None

        if result != 0:
            print(f"Servo {servo_id} comm error: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            print(f"Servo {servo_id} packet error: {self.packetHandler.getRxPacketError(error)}")

        # Return the correct tuple of values based on the operation
        return (read_value, result, error) if not write else (result, error)
    
    def set_torque(self, servo_id, enable):
        """Enables or disables torque on a servo."""
        state = "enabled" if enable else "disabled"
        if self._execute_servo_command(servo_id, ADDR_TORQUE_ENABLE, 1 if enable else 0):
            print(f"Torque {state} on Servo {servo_id}")

    def read_position(self, servo_id):
        """Reads the current position of a servo."""
        # Change size to 2 and properly unpack the returned values
        raw_position, result, error = self._execute_servo_command(
            servo_id, ADDR_PRESENT_POSITION, None, size=2, write=False
        )

        # Check if the read was successful before returning the value
        if result == 0 and raw_position is not None:
            return raw_position
        else:
            return -1 # Return -1 on failure
        
    def move_servo(self, servo_id, position, speed=800, acc=50):
        """Moves a servo to a specific position with given speed and acceleration."""
        self._execute_servo_command(servo_id, ADDR_GOAL_ACC, acc, size=1)
        self._execute_servo_command(servo_id, ADDR_GOAL_SPEED, speed, size=2)
        self._execute_servo_command(servo_id, ADDR_GOAL_POSITION, position, size=4)

    def set_midpoint_as_current_position(self, servo_id):
        """Sets the servo's current position as its new midpoint (zero offset)."""
        print(f"Setting midpoint for Servo {servo_id}...")
        self.set_torque(servo_id, False)
        time.sleep(0.1)  # Small delay
        # Command 128 resets the offset to the current position
        if self._execute_servo_command(servo_id, ADDR_MIDPOINT_OFFSET, 128):
             print(f"Midpoint for Servo {servo_id} has been set to its current position.")
        self.set_torque(servo_id, True)

    @staticmethod
    def circular_diff(current, base, max_val=4096):
        """Calculates the shortest difference in a circular space (e.g., servo positions)."""
        diff = current - base
        if diff > max_val / 2:
            diff -= max_val
        elif diff < -max_val / 2:
            diff += max_val
        return diff