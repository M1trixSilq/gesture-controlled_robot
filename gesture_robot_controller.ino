#include <Servo.h>

//Пины драйвера моторов (L298N)
const uint8_t LEFT_IN1 = 4;
const uint8_t LEFT_IN2 = 5;
const uint8_t LEFT_PWM = 6;   // PWM

const uint8_t RIGHT_IN1 = 7;
const uint8_t RIGHT_IN2 = 8;
const uint8_t RIGHT_PWM = 10; // PWM

//Сервопривод захвата
const uint8_t GRIP_SERVO_PIN = 9;
const int GRIP_OPEN_ANGLE = 20;
const int GRIP_CLOSE_ANGLE = 95;

//Скорости шасси
const uint8_t DRIVE_SPEED = 190;
const uint8_t TURN_SPEED = 180;

// Если команды пропали, робот автоматически останавливается.
const unsigned long COMMAND_TIMEOUT_MS = 700;

Servo gripServo;
String serialBuffer;
unsigned long lastCommandTime = 0;

void setup() {
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(LEFT_PWM, OUTPUT);

  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);

  gripServo.attach(GRIP_SERVO_PIN);
  gripServo.write(GRIP_OPEN_ANGLE);

  stopMotors();

  Serial.begin(115200);
  Serial.println("[Robot] Ready. Waiting for commands...");
  Serial.println("[Robot] Supported: 0000/0001/0010/0100/1000, GRIP_OPEN, GRIP_CLOSE");

  serialBuffer.reserve(32);
  lastCommandTime = millis();
}

void loop() {
  readSerialCommands();

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    stopMotors();
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 31) {
        serialBuffer = "";
      }
    }
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  lastCommandTime = millis();

  // Движение
  if (cmd == "0100" || cmd == "FORWARD") {
    driveForward(DRIVE_SPEED);
    Serial.println("[Robot] FORWARD");
    return;
  }

  if (cmd == "1000" || cmd == "BACKWARD") {
    driveBackward(DRIVE_SPEED);
    Serial.println("[Robot] BACKWARD");
    return;
  }

  if (cmd == "0001" || cmd == "TURN_RIGHT") {
    turnRightOnSpot(TURN_SPEED);
    Serial.println("[Robot] TURN_RIGHT (on spot)");
    return;
  }

  if (cmd == "0010" || cmd == "TURN_LEFT") {
    turnLeftOnSpot(TURN_SPEED);
    Serial.println("[Robot] TURN_LEFT (on spot)");
    return;
  }

  if (cmd == "0000" || cmd == "STOP") {
    stopMotors();
    Serial.println("[Robot] STOP");
    return;
  }

  // Захват манипулятора
  if (cmd == "GRIP_OPEN") {
    gripServo.write(GRIP_OPEN_ANGLE);
    Serial.println("[Robot] GRIP_OPEN");
    return;
  }

  if (cmd == "GRIP_CLOSE") {
    gripServo.write(GRIP_CLOSE_ANGLE);
    Serial.println("[Robot] GRIP_CLOSE");
    return;
  }

  Serial.print("[Robot] Unknown command: ");
  Serial.println(cmd);
}

void driveForward(uint8_t speedValue) {
  setMotor(LEFT_IN1, LEFT_IN2, LEFT_PWM, speedValue);
  setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, speedValue);
}

void driveBackward(uint8_t speedValue) {
  setMotor(LEFT_IN1, LEFT_IN2, LEFT_PWM, -speedValue);
  setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, -speedValue);
}

void turnLeftOnSpot(uint8_t speedValue) {
  setMotor(LEFT_IN1, LEFT_IN2, LEFT_PWM, -speedValue);
  setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, speedValue);
}

void turnRightOnSpot(uint8_t speedValue) {
  setMotor(LEFT_IN1, LEFT_IN2, LEFT_PWM, speedValue);
  setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, -speedValue);
}

void stopMotors() {
  setMotor(LEFT_IN1, LEFT_IN2, LEFT_PWM, 0);
  setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, 0);
}

void setMotor(uint8_t in1, uint8_t in2, uint8_t pwmPin, int speedSigned) {
  int pwm = constrain(abs(speedSigned), 0, 255);

  if (speedSigned > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else if (speedSigned < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }

  analogWrite(pwmPin, pwm);
}
