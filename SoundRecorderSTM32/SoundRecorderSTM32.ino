#define bufflength 8500
float prev1;
float prev2; 
float prev3;
float prev4;
float pres;
unsigned int timeout=bufflength+1;
unsigned int readValue;
float localAvg;
int timestamp=0;
byte buff[bufflength*2];

void setup(){
  Serial.begin(38400);
  pres=2800;prev1=pres;prev2=prev1;prev3=prev2;prev4=prev3;
}

void loop(){
  readValue=analogRead(PA0);
  localAvg=(prev1+prev2+prev3+prev4)/4;
  pres=readValue*0.1+pres*0.9;
  //Serial.println(pres);
  if(abs(pres-localAvg)>5){
    if(timeout>bufflength){
      timeout=bufflength;
      
      timestamp=millis();
    }
  }
  if(timeout<bufflength+1){
    buff[(bufflength-timeout)*2]=highByte((unsigned int)prev4);
    buff[(bufflength-timeout)*2+1]=lowByte((unsigned int)prev4);
    timeout--;
    if (timeout==0){
      Serial.println(millis()-timestamp);
      timestamp=millis();
      //for(int i=0;i<bufflength*2-1;i+=2){
//        Serial.println(buff[i]*256+buff[i+1]);
      //}
      
      //Serial.write('\n');Serial.write('\n');
      //Serial.write(buff,bufflength*2);
      //byte temp[8] ={highByte(prev3),lowByte(prev3),highByte(prev2),lowByte(prev2),highByte(prev1),lowByte(prev1),highByte(readValue),lowByte(readValue)};
      //Serial.write(temp,8);
      //Serial.write('\n');Serial.write('\n');
      timeout=-1;
    }
  }
  prev4=prev3;
  prev3=prev2;
  prev2=prev1;
  prev1=pres;
}
