#define bufflength 890
unsigned int prev1;
unsigned int prev2; 
unsigned int prev3;
unsigned int prev4;
unsigned int timeout=bufflength+1;
unsigned int readValue;
unsigned int localAvg;
int timestamp=0;
byte buff[bufflength*2];

void setup(){
  Serial.begin(9600);
  readValue=826;prev1=readValue;prev2=prev1;prev3=prev2;prev4=prev3;
}

void loop(){
  readValue=analogRead(A6);
  localAvg=(prev1+prev2+prev3+prev4)/4;
  if(abs((int)(readValue-localAvg))>5){
    if(timeout>bufflength){
      timeout=bufflength;
      timestamp=millis();
    }
  }
  //Serial.println(timeout);
  //Serial.println(timeout<bufflength+1);
  Serial.println(timeout<bufflength+1);
  if(timeout<bufflength+1){
    buff[(bufflength-timeout)*2]=highByte(prev4);
    buff[(bufflength-timeout)*2+1]=lowByte(prev4);
    timeout--;
    if (timeout==0){
      //Serial.write('\n');Serial.write('\n');
      //Serial.println(millis()-timestamp);
      timestamp=millis();
      //for(int i=0;i<bufflength*2-1;i+=2){
      //  Serial.println(buff[i]*256+buff[i+1]);
      //}
      //Serial.write(buff,bufflength*2);
      //byte temp[8] ={highByte(prev3),lowByte(prev3),highByte(prev2),lowByte(prev2),highByte(prev1),lowByte(prev1),highByte(readValue),lowByte(readValue)};
      //Serial.write(temp,8);
      //Serial.write('\n');
      timeout=-1;
      delay(500);
    }
  }
  prev4=prev3;
  prev3=prev2;
  prev2=prev1;
  prev1=readValue;
}
