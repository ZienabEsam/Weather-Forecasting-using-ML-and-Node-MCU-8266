#include <WiFiClient.h>
#include <ESP8266WiFi.h>
#include "DHT.h"
#define DHTPIN D4
#define DHTTYPE DHT11
DHT myDHT(D4,DHTTYPE);
WiFiClient client;
String APIKEY = "LRP0WI5RLNX7BY3B";
const char* thingspeak = "api.thingspeak.com";
const char* Network_Name = "TP-LINK_1904";
const char* PASSWORD = "42060222";
int port = 80;

void setup()
{
 Serial.begin(115200);
 myDHT.begin();
 //wifi set up
 WiFi.mode(WIFI_STA);
delay(1000);
WiFi.begin(Network_Name,PASSWORD);
delay(3000);
Serial.print("Connecting to Wifi please wait...");
while(WiFi.status() != WL_CONNECTED)
{
  Serial.print(".");
  delay(700);
  }
Serial.println("Connected to TP-LINK_1904");
delay(6000);
}

void loop()
{
 
Serial.print("DHT11 is getting ready");
delay(2000);
//reading temperature from sensor
float t = myDHT.readTemperature();
//reading humedity z
float h = myDHT.readHumidity();

Serial.println("Temperature is");
Serial.println(t);
Serial.println("Humidity now is");
Serial.println(h);
//connecting to thing speak and posting results
///////////////////////////////////////////////////////
Serial.println("Connecting to website to post results");
while(!client.connect(thingspeak,port))
{
 Serial.print("trying to connect to Thing speak");
 Serial.print(".");
 delay(700);
  }
  
  Serial.println("Sucessfully connected to host" + String(thingspeak));
  while(client.connect(thingspeak,port))
  {
    String postStr = APIKEY ;
    postStr += "&field1=";
    postStr += String(t);
    postStr += "&field2=";
    postStr += String(h);
    postStr += "\r\n\r\n";
    Serial.println("- sending data...");
    client.print("POST /update HTTP/1.1\n");
    client.print("Host: api.thingspeak.com\n");
    client.print("Connection: close\n");
    client.print("X-THINGSPEAKAPIKEY: " + APIKEY + "\n");
    client.print("Content-Type: application/x-www-form-urlencoded\n");
    client.print("Content-Length: ");
    client.print(postStr.length());
    client.print("\n\n");
    client.print(postStr);
  }
  client.stop();
delay(60000);
}
