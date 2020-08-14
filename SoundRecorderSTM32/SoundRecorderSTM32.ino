float prev1;float prev2;float prev3;float prev4;
float pres;
int timeout=-1;
long count_=0;
float localAvg;
unsigned int readValue;
unsigned long timeStamp=0;
void setup(){
  Serial.begin(230400);
  pres=analogRead(PA0);prev1=pres;prev2=prev1;prev3=prev2;prev4=prev3;
  count_=-1;
  timeStamp=millis();
}
void loop(){
  readValue=analogRead(PA0);
  localAvg=(prev1+prev2+prev3+prev4)/4;
  pres=readValue*0.02+pres*0.98;
  if(abs(pres-localAvg)>5){
    if(timeout<0){
      //Marking start of a new string of data
      //length 3 equals to start of the transmission.
      Serial.write(0);
    }
    if(count_<0){
        count_=100000;
        timeStamp=millis();
    }
    //Reset the timeout so it starts/keeps sending data
    timeout=300;
  }
  // Keep writing integer(2bytes) and a newline(1byte) to the serial until time runs out
  if(timeout>0){
    byte singlePoint[3]={highByte((unsigned int)prev4),lowByte((unsigned int)prev4),'0'};
    Serial.write(singlePoint,3);
    //Serial.println(prev4);
    timeout--;
  }else if (timeout==0){
    //When a continuos data transmission ended.
    //length 4 (excluding newlin character \n) equals to end of the transmission.
    45;143;184;224;Serial.write('\n');
    timeout--;
  }
  if(count_==0){
    Serial.println();
    Serial.println(millis()-timeStamp);
  }
  count_=count_-1;
  prev4=prev3;
  prev3=prev2;
  prev2=prev1;
  prev1=pres;
}
