#include <Adafruit_NeoPixel.h>
#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <WiFi.h>

#ifndef LED_PIN
#define LED_PIN 5
#endif

constexpr char TOPIC_PREFIX[] = "lightsense";
constexpr size_t MAX_MESSAGE_BYTES = 512;

Adafruit_NeoPixel strip(PIXEL_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
char commandTopic[80];

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.printf("\nWi-Fi connected: %s\n", WiFi.localIP().toString().c_str());
}

void applyCommand(const byte *payload, unsigned int length) {
  if (length == 0 || length >= MAX_MESSAGE_BYTES) {
    Serial.println("Rejected empty or oversized command");
    return;
  }

  JsonDocument document;
  DeserializationError error = deserializeJson(document, payload, length);
  if (error) {
    Serial.printf("Invalid JSON: %s\n", error.c_str());
    return;
  }

  JsonArray pixels = document["pixels"].as<JsonArray>();
  if (pixels.size() != PIXEL_COUNT) {
    Serial.printf("Expected %d pixels, received %d\n", PIXEL_COUNT, pixels.size());
    return;
  }

  const int brightness = constrain(document["brightness"] | 255, 0, 255);
  strip.setBrightness(brightness);
  for (size_t index = 0; index < PIXEL_COUNT; ++index) {
    const int red = constrain(pixels[index]["r"] | 0, 0, 255);
    const int green = constrain(pixels[index]["g"] | 0, 0, 255);
    const int blue = constrain(pixels[index]["b"] | 0, 0, 255);
    strip.setPixelColor(index, strip.Color(red, green, blue));
  }
  strip.show();
}

void onMqttMessage(char *topic, byte *payload, unsigned int length) {
  if (strcmp(topic, commandTopic) == 0) {
    applyCommand(payload, length);
  }
}

void connectMqtt() {
  while (!mqttClient.connected()) {
    String clientId = String("lightsense-") + LOCATION_TOPIC + "-" +
                      String(static_cast<uint32_t>(ESP.getEfuseMac()), HEX);
    if (mqttClient.connect(clientId.c_str())) {
      mqttClient.subscribe(commandTopic, 1);
      Serial.printf("Subscribed to %s\n", commandTopic);
    } else {
      Serial.printf("MQTT failed (%d); retrying\n", mqttClient.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.clear();
  strip.show();

  snprintf(commandTopic, sizeof(commandTopic), "%s/%s/set", TOPIC_PREFIX,
           LOCATION_TOPIC);
  connectWifi();
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  mqttClient.setCallback(onMqttMessage);
  mqttClient.setBufferSize(MAX_MESSAGE_BYTES);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }
  if (!mqttClient.connected()) {
    connectMqtt();
  }
  mqttClient.loop();
}

