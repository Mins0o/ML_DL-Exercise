int prev1;int prev2;int prev3;int prev4;
int timeout=-1;
long count=0;
unsigned long timeStamp=0;
void setup(){
  Serial.begin(9600);
  pinMode(13,OUTPUT);
  int readValue=analogRead(A6);prev1=readValue;prev2=prev1;prev3=prev2;prev4=prev3;
  count=-1;
  timeStamp=millis();
}
void loop(){
  int readValue=analogRead(A6);
  int localAvg=(prev1+prev2+prev3+prev4)/4;
  if(abs(readValue-localAvg)>5){
    if(timeout<0){
      //Marking start of a new string of data
      //length 3 equals to start of the transmission.
      //Serial.write(0);
      if(count<0){
        count=10000;
        timeStamp=millis();
      }
    }
    //Reset the timeout so it starts/keeps sending data
    timeout=300;
    digitalWrite(13,LOW);
  }else{digitalWrite(13,HIGH);}
  // Keep writing integer(2bytes) and a newline(1byte) to the serial until time runs out
  if(timeout>0){
    byte singlePoint[3]={highByte(prev4),lowByte(prev4),'\n'};
    //Serial.write(singlePoint,3);
    //Serial.println(prev4);
    timeout--;
  }else if (timeout==0){
    //When a continuos data transmission ended.
    //length 4 (excluding newlin character \n) equals to end of the transmission.
    //Serial.write(0);Serial.write(0);Serial.write(0);Serial.write(0);Serial.write('\n');
    timeout--;
  }
  if(count==0){
    Serial.println(millis()-timeStamp);
  }
  count--;
  prev4=prev3;
  prev3=prev2;
  prev2=prev1;
  prev1=readValue;
}
