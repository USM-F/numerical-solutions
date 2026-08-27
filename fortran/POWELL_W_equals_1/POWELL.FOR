      SUBROUTINE POWELL(FUN,STEP,ACC)                                         
      implicit none
      real*8 x,y,s,fx,fy,direct,dum,before,first,w,secnd,
     *stepa,step,f1,f2,f3,sum,a,flag,acc,ffirst,fsecnd
      integer 
     *n,nfunct,ndrv,iter,indic,iprint,iconvg,ntry,n1,i,j,k,isave,ii
      COMMON/OPTIM1/X(150),Y(150),S(150),FX,FY                                   
      COMMON/OPTIM2/DIRECT(150,150),DUM(150),BEFORE(150),FIRST(150)                
      COMMON/OPTIM3/N,NFUNCT,NDRV,ITER,INDIC,IPRINT                           
      DIMENSION W(150),SECND(150)                                               
      EQUIVALENCE(W,SECND)                                                    
      external fun
C     N-×ÈÑËO ÏEPEMEHHÛX                                                      
C     ICONVG -                                                                
C     STEP - HA×AËÜHÛÉ PAÇMEP ØAÃA                                            
C     ACC - TREÁÓEMAß TO×HOCTÜ                                                
      INDIC=2
      ICONVG=1
      ITER=0 
      NTRY=1 
      N1=N-1                           
      STEPA=STEP                                                              
C     SET UP THE INITIAL DIRECTION MATRIX(USING UNIT VECTORS)                 
      do i=1,n                                                              
       do j=1,n                                                              
        direct(j,i)=0.                                                          
       enddo
       direct(i,i)=1.                                                          
      enddo
 100  CALL FUN(X,FX)                                                          
      IF(IPRINT.EQ.1) PRINT 200,ITER,NFUNCT,FX,(X(I),I=1,N)                    
      GO TO 301                                                               
C                                                                             
   3  ITER=ITER+1                                                             
 301  F1=FX                                                                   
      do i=1,n                                                              
       before(i)=x(i)                                                          
      enddo
      SUM=0.                                                                  
C                                                                             
      do i=1,n                                                              
       do j=1,n                                                              
        s(j)=direct(j,i)*step                                                   
       enddo
C                                                                             
       CALL SEARCH(FUN)                                                        
       IF(IPRINT.EQ.1) PRINT 200,ITER,NFUNCT,FX,(X(K),K=1,N)                   
       A=FX-FY                                                                 
       IF(A-SUM) then                                                          
        ISAVE=I                                                                 
        SUM=A                                                                   
       endif
C                                                                             
       do j=1,n                                                              
        x(j)=y(j)                                                               
       enddo
       FX=FY                                                                   
      enddo
C                                                                             
      F2=FX                                                                   
      do i=1,n                                                             
       w(i)=2.0*x(i)-before(i)                                                 
      enddo
      CALL FUN(W,F3)                                                          
      A=F3-F1                                                                 
      IF(A) 11,19,19                                                           
  11  A=2.*(F1-2.*F2+F3)*((F1-F2-SUM)/A)**2                                   
      IF(A-SUM) 12,19,19                                                       
C                                                                             
  12  IF(ISAVE-N) 13,15,15                                                     
  13  CONTINUE
      do i=isave,n1                                                        
       ii=i+1                                                                  
       do j=1,n                                                             
        direct(j,i)=direct(j,ii)
       enddo
      enddo
C                                                                             
  15  A=0.                                                                    
      do j=1,n                                                             
       direct(j,n)=x(j)-before(j)                                              
       a=direct(j,n)**2+a                                                      
      enddo
      A=1./SQRT(A)                                                            
      do j=1,n                                                             
       direct(j,n)=direct(j,n)*a                                               
       s(j)=direct(j,n)*step                                                   
      enddo
      CALL SEARCH(FUN)                                                        
      FX=FY                                                                   
      do i=1,n                                                             
       x(i)=y(i)                                                               
      enddo
C     TECT  CÕOÄÈMOCTÈ                                                     
 19   CALL CONVPL(F1,FX,BEFORE,X,FLAG,N,ACC)                                  
      IF(FLAG) 22,22,20                                                        
C                                                                             
  20  IF(F1-FX) 121,120,120                                                    
 121  STEP=-0.4*SQRT(ABS(F1-FX))                                              
      GO TO 123                                                               
 120  STEP=0.4*SQRT(F1-FX)                                                    
 123  IF(STEPA-STEP) 21,3,3                                                    
  21  STEP=STEPA                                                              
      GO TO 3                                                                 
C                                                                             
  22  GO TO(23,24),ICONVG                                                     
  23  RETURN                                                                  
  24  GO TO(25,27),NTRY                                                       
  25  NTRY=2                                                                  
      FIRST(I)=X(I)                                                           
      do i=1,n                                                             
       x(i)=x(i)+acc*10.                                                       
      enddo
      FFIRST=FX                                                               
      GO TO 100                                                               
C                                                                             
  27  FSECND=FX                                                               
      A=0.                                                                    
      do i=1,n                                                             
       secnd(i)=x(i)                                                           
       s(i)=first(i)-secnd(i)                                                  
       a=a+s(i)**2                                                             
      enddo
      IF(A) 23,23,29                                                           
  29  A=STEP/SQRT(A)                                                          
      do i=1,n                                                             
       s(i)=s(i)*a                                                             
      enddo
      CALL SEARCH(FUN)                                                        
C                                                                             
      CALL CONVPL(FFIRST,FY,FIRST,Y,FLAG,N,ACC)                               
      IF(FLAG) 32,32,31                                                        
31    CALL CONVPL(FSECND,FY,SECND,Y,FLAG,N,ACC)                               
      IF(FLAG) 32,32,34                                                        
  32  CONTINUE
      do i=1,n                                                             
       x(i)=y(i)                                                               
      enddo
      FX=FY                                                                   
      RETURN                                                                  
C                                                                             
  34  A=A/STEP                                                                
      do i=1,n                                                             
       direct(i,1)=(first(i)-secnd(i))*a                                       
       first(i)=secnd(i)                                                       
      enddo
      GO TO 3                                                                 
 200  FORMAT(2X,2I7,E12.5,/,10E10.3)                                          
      END                                                           

             
      SUBROUTINE CONVPL(FI,FF,RI,RF,FLAG,N,ACC)                               
      implicit none
      integer i,n
      real*8 fi,ff,ri,rf,flag,acc
      DIMENSION RI(150),RF(150)                                                 
      FLAG=2.                                                                 
      IF(ABS(FI)-ACC) 2,2,1                                                    
   1  IF(ABS((FI-FF)/FI)-ACC) 3,3,7                                            
   2  IF(ABS(FI-FF)-ACC) 3,3,7                                                 
   3  CONTINUE
      DO 6 I=1,N                                                              
      IF(ABS(RI(I))-ACC) 5,5,4                                                 
   4  IF(ABS((RI(I)-RF(I))/RI(I))-ACC) 6,6,7                                   
   5  IF(ABS(RI(I)-RF(I))-ACC) 6,6,7                                           
   6  CONTINUE                                                                
      FLAG=-2.                                                                
   7  RETURN                                                                  
      END                                                        
      
      SUBROUTINE SEARCH(FUN)                                                  
C     COGGIN METHOD OF UNIDIMENSIONAL SEARCH                                   
      implicit none
      external fun
      integer n,iexit,ntol,nfunct,ndrv,iter,indic,iprint,k,m,i
      real*8 x,y,s,fx,fy,h,delx,delg,gx,ftol,ftol2,fa,fb,fc,
     *da,db,dc,step,d,dxnorm,snorm,ratio,f,a
      COMMON/OPTIM1/X(150),Y(150),S(150),FX,FY                                   
      COMMON/OPTIM2/H(150,150),DELX(150),DELG(150),GX(150)                         
      COMMON/OPTIM3/N,NFUNCT,NDRV,ITER,INDIC,IPRINT                           
      IEXIT=0.
      NTOL=0                                                       
      FTOL=1.E-3
      FTOL2=1.E-5                                                
      FA=FX
      FB=FX
      FC=FX                                                   
      DA=0.
      DB=0.
      DC=0.                                                   
      K=-2
      M=0
      STEP=1.
      D=STEP                                           
C       USE THE PARAMETER INDIC TO INDICATE NOW THE SEARCH VECTOR
C       LENGTH SHOULD BE SCALED.
C       INDIC=2 DO NOT SCALE.TAKE LENGTH GIVEN BY MIN CALCULATION
C       INDIC=1 SCALE ONLY IF THE LENGTH OF THE LAST STEP WAS SHORTER THAN
C       THE LENGTH OF THE SEARCH VECTOR. SCALE TO LENGTH OF LAST STEP.
C       INDIC=ANYTHING BUT 1 OR 2 RESULTS IN SCALING TO LENGTH OF LAST STEP.
      IF(INDIC.EQ.2.OR.ITER.EQ.0) GO TO 1                                     
C       FIND NORM 0F S AND DELX 
      DXNORM=0.
      SNORM=0.                                                    
      DO 102 I=1,N                                                            
      DXNORM=DXNORM+DELX(I)**2                                                
 102  SNORM=SNORM+S(I)**2                                                     
      IF(INDIC.EQ.1.AND.DXNORM.GE.SNORM) GO TO 1                              
      RATIO=DXNORM/SNORM                                                      
      STEP=SQRT(RATIO)
      D=STEP                                                 
C       START THE SEARCH THE BOUND OF MIN   
   1  CONTINUE
      DO 2 I=1,N                                                              
   2  Y(I)=X(I)+D*S(I)                                                        
      CALL FUN(Y,F)                                                           
      K=K+1                                                                   
      IF(F-FA) 5,3,6                                                          
C       NO CHANGE IN FUNCTION VALUE.
   3  CONTINUE
      DO 4 I=1,N                                                              
   4  Y(I)=X(I)+DA*S(I)                                                       
      FY=FA                                                                   
C       SEARCH FAILED. 
      IF(IPRINT.EQ.1) PRINT 210                                               
 210  FORMAT(2X,'SEARCH,FAILED')                                              
      GO TO 326                                                               
C     THE FUNCTION IS STILL DECREASING INCREASE THE STEP SIZE BU DOUBLE       
   5  FC=FB
      FB=FA
      FA=F                                                    
      DC=DB
      DB=DA
      DA=D                                                    
      D=2.*D+STEP                                                             
      GO TO 1                                                                 
C        MINIMUM IS BOUNDED IN AT LEAST ONE DIRECTION.  
   6  IF(K) 7,8,9                                                              
   7  FB=F                                                                    
      DB=D
      D=-D
      STEP=-STEP                                                
      GO TO 1                                                                 
   8  FC=FB                                                                   
      FB=FA
      FA=F
      DC=DB
      DB=DA
      DA=D                                     
      GO TO 21                                                                
   9  DC=DB                                                                   
      DB=DA
      DA=D
      FC=FB
      FB=FA
      FA=F                                     
  10  D=0.5*(DA+DB)                                                           
      DO 11 I=1,N                                                             
  11  Y(I)=X(I)+D*S(I)                                                        
      CALL FUN(Y,F)                                                           
  12  IF((DC-D)*(D-DB)) 15,13,18                                               
  13  CONTINUE
      DO 14 I=1,N                                                             
  14  Y(I)=X(I)+DB*S(I)                                                       
      FY=FB                                                                   
      IF(IEXIT.EQ.1) GO TO 32                                                 
C    SEARCH FAILED                                                            
      IF(IPRINT.EQ.1) PRINT 210                                               
      GO TO 325                                                               
  15  IF(F-FB) 16,13,17                                                        
  16  FC=FB                                                                   
      FB=F
      DC=DB
      DB=D                                                     
      GO TO 21                                                                
17    FA=F                                                                    
      DA=D                                                                    
      GO TO 21                                                                
  18  IF(F-FB) 19,13,20                                                        
  19  FA=FB                                                                   
      FB=F
      DA=DB
      DB=D                                                     
      GO TO 21                                                                
  20  FC=F                                                                    
      DC=D                                                                    
C     PERFORM THE PARABOLIC INTERPOLATION                                     
  21  A=FA*(DB-DC)+FB*(DC-DA)+FC*(DA-DB)                                      
      IF(A) 22,30,22                                                           
  22  D=0.5*((DB*DB-DC*DC)*FA+(DC*DC-DA*DA)*FB+(DA*DA-DB*DB)*FC)/A            
      IF((DA-D)*(D-DC)) 13,13,23                                               
  23  CONTINUE
      DO 24 I=1,N                                                             
  24  Y(I)=X(I)+D*S(I)                                                        
      CALL FUN(Y,F)                                                           
C     CHECK FOR CONVERGENCE.IF NOT ACHEIVED,RECYCLE.                          
      IF(ABS(FB)-FTOL2) 25,25,26                                               
  25  A=1.                                                                    
      GO TO 27                                                                
  26  A=1./FB                                                                 
  27  IF((ABS(FB-F)*A)-FTOL) 28,28,12                                          
C     CONVERGENCE ACHEIVED.RETURN WITHTHE SMALLER OF F AND FB.                
  28  IEXIT=1                                                                 
      IF(F-FB) 29,13,13                                                        
  29  FY=F                                                                    
      GO TO 32                                                                
  30  IF(M) 31,31,13                                                           
  31  M=M+1                                                                   
      GO TO 10                                                                
  32  CONTINUE
      DO 99 I=1,N                                                             
      IF(Y(I).NE.X(I)) GO TO 325                                              
  99  CONTINUE                                                                
      GO TO 33                                                                
 325  CONTINUE                                                                
 326  IF(FY.LT.FX) RETURN                                                     
      IF((S(1).NE.-GX(1)).OR.(FY.LT.FX)) RETURN                                
      if(iprint.eq.1) PRINT 211                                                               
  211 FORMAT(2X,'Håóäà÷íûé ïîèñê')                                            
      RETURN                                                                  
  33  IF(NTOL.EQ.5) GO TO 34                                                   
      IEXIT=0                                                                 
      NTOL=NTOL+1                                                             
      FTOL=FTOL/10.                                                           
      GO TO 12                                                                
  34  RETURN                                                                  
      END

