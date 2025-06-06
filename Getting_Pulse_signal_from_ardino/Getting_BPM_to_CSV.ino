
/*  Getting_BPM_to_Monitor prints the BPM to the Serial Monitor, using the least lines of code and PulseSensor Library.
 *  Tutorial Webpage: https://pulsesensor.com/pages/getting-advanced
 *
--------Use This Sketch To------------------------------------------
1) Displays user's live and changing BPM, Beats Per Minute, in Arduino's native Serial Monitor.
2) Learn about using a PulseSensor Library "Object".
3) Blinks the builtin LED with user's Heartbeat.
--------------------------------------------------------------------*/
#include <PulseSensorPlayground.h>     // Includes the PulseSensorPlayground Library. 
struct Packet{
  unsigned long Signal;                // holds the incoming raw data. Signal value can range from 0-1024
  unsigned long time;
  unsigned long funnyno;

};

const int PulseWire = 0;       // PulseSensor PURPLE WIRE connected to ANALOG PIN 0
const int LED = LED_BUILTIN;          // The on-board Arduino LED, close to PIN 13.
int Threshold = 550;           // Determine which Signal to "count as a beat" and which to ignore.
                               // Use the "Gettting Started Project" to fine-tune Threshold Value beyond default setting.
                               // Otherwise leave the default "550" value. 
PulseSensorPlayground pulseSensor;  // Creates an instance of the PulseSensorPlayground object called "pulseSensor"


void setup() {   

  Serial.begin(115200);          // For Serial Monitor

  // Configure the PulseSensor object, by assigning our variables to it. 
  pulseSensor.analogInput(PulseWire);   
  pulseSensor.blinkOnPulse(LED);       //auto-magically blink Arduino's LED with heartbeat.
  pulseSensor.setThreshold(Threshold);   
  // Double-check the "pulseSensor" object was created and "began" seeing a signal. 
  if (pulseSensor.begin()) {
    Serial.println("We created a pulseSensor Object !");  //This prints one time at Arduino power-up,  or on Arduino reset.  
  }
}


void loop() {
  
  Packet p;
  p.Signal = analogRead(PulseWire);  // Read the PulseSensor's value.
  p.time = millis();
  p.funnyno = 4222;
  Serial.write((byte*)&p, sizeof(p));     // sends 12 bytes
  delay(20);       
}