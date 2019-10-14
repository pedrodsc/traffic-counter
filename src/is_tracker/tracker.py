import numpy as np
from numpy.linalg import inv


class Kalman(object):
  def __init__(self,x,z,P,A,H,R,Q):
    
    self.x, self.z = x, z
    
    self.P, self.A, self.H = P, A, H
    
    self.R, self.Q = R, Q

  def predict(self):
    
    self.x = self.A.dot(self.x) # (C.5)
    self.P = self.A.dot(self.P).dot(self.A.T) + self.Q # (C.6)
  
  def update(self,z):
    
    S = self.H.dot(P).dot(self.H.T) + self.R # (C.9)
    K = self.P.dot(self.H.T).dot(inv(S)) # (C.9)
    y = z - self.H.dot(self.x) # (C.10)
    self.x = self.x + K.dot(y) # (C.10)
    self.P = self.P - K.dot(self.H).dot(self.P) # (C.11)
    
    return self.x,self.P

class Tracker():
    def __init__(self):
        # Lista, vetor ou dicionário de objetos sendo trackeados
        # Numero de objetos trackeados
        # Hiperparâmetros: Limite de frames para remover um objeto, incertezas*
        
        # Não sei se as incertezas serão para cada objeto ou global
    
    def update(self, detectedObjects): ## Vai se chamar update?
        # Associa os novos dados aos objetos antigos
        # Roda o update de todos os objetos detectados
        # Adiciona novos objetos
    
