import numpy as np
from numpy.linalg import inv
from collections import OrderedDict
from scipy.spatial import distance as dist

class TrackedObject(object):
    def __init__(self,object_ID,class_name,score,x_pos,y_pos):
        self.class_name = class_name
        self.object_ID = object_ID
        self.score = 0
        self.missing_frames = 0
        self.missing = False
        # Covariance matrix
        self.P = np.eye(6)
        self.x = np.array([x_pos,y_pos,0,0,0,0],dtype=np.float)
        self.z = self.x[0:2]

class Kalman(object):
  def __init__(self,R_var=0.5,Q_var=0.1,dt=0.1):
    
    self.R = np.eye(2)*R_var # Measure noise
    self.Q = np.eye(6)*Q_var # Process noise
    
    # Matrix to extract Cx and Cy from vector x
    self.H = np.array([[1., 0, 0, 0, 0, 0],[0., 1., 0, 0, 0, 0]])
    # Model of the process
    self.A = np.array( [[1., 0, dt,  0, (dt**2)/2, 0],    # Cx = Cx + Vx*dt + 0.5*Ax*dt**2
                        [0, 1,  0, dt, 0,     (dt**2)/2], # Cy = Cy + Vy*dt + 0.5*Ay*dt**2
                        [0, 0,  1,  0, dt,    0],         # Vx = Vx + Ax*dt
                        [0, 0,  0,  1, 0,     dt],        # Vy = Vy + Ay*dt
                        [0, 0,  0,  0, 1,     0],         # Ax = Ax
                        [0, 0,  0,  0, 0,     1]])        # Ay = Ay

  def predict(self,tracked_object):
    
    tracked_object.x = self.A.dot(tracked_object.x) # (C.5)
    tracked_object.P = self.A.dot(tracked_object.P).dot(self.A.T) + self.Q # (C.6)
  
  def update(self,tracked_object):
    
    S = self.H.dot(tracked_object.P).dot(self.H.T) + self.R # (C.9)
    K = tracked_object.P.dot(self.H.T).dot(inv(S)) # (C.9)
    y = tracked_object.z - self.H.dot(tracked_object.x) # (C.10)
    tracked_object.x = tracked_object.x + K.dot(y) # (C.10)
    tracked_object.P = tracked_object.P - K.dot(self.H).dot(tracked_object.P) # (C.11)

class Tracker(object):
    def __init__(self, img_shape, max_missing_frames = 15, max_radius = 80):
        # Lista, vetor ou dicionário de objetos sendo trackeados
        self.tracked_objects = OrderedDict()
        # Numero de objetos já trackeados
        self.next_obj_ID = 1
        # Número de objetos sendo trackeados
        self.objects_being_tracked = 0
        # Hiperparâmetros: Limite de frames para remover um objeto, incertezas*
        self.max_missing_frames = max_missing_frames
        # Não sei se as incertezas serão para cada objeto ou global
        
        # Distancia maxima pra tentar associar objetos
        self.max_radius = max_radius
        # This matrix times bbox(0 to 1) gives the centroid of the bbox
        (w,h) = img_shape
        self.centroid_matrix = np.array([[w/2,0,w/2,0],[0,h/2,0,h/2]],dtype=np.float)

        self.kalman = Kalman()
        
    
    def register(self, class_name, score, centroid):
        x_pos = centroid[0]
        y_pos = centroid[1]
        self.tracked_objects[self.next_obj_ID] = TrackedObject(self.next_obj_ID,class_name,score,x_pos,y_pos)
        self.next_obj_ID += 1
        self.objects_being_tracked += 1
        
    
    def deregister(self, object_ID):
        self.tracked_objects.pop(object_ID)
        self.objects_being_tracked -= 1
    
    def dataAssociation(self, detected_objects):
        # Associa os novos dados aos objetos antigos
        # Adiciona novos objetos
        
        # Parse detected_objects
        boxes, scores, classes, nums = detected_objects
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        # bbox, confidence, class name, number of detected objects
        
        # centroids in x,y format
        centroids = np.empty((nums,2))      
        
        for i in range(nums):
            box = boxes[i]
            #                               |bx1|
            #  |Cx| = |w/2  0  w/2 0  |  *  |by1|
            #  |Cy|   | 0  h/2  0 h/2 |     |bx2|
            #                               |by2|
            centroids[i] = self.centroid_matrix.dot(box)

        if self.objects_being_tracked == 0:
            for i in range(nums):
                self.register(classes[i],scores[i],centroids[i])
        else:
            # predict nos kalmans
            for key in self.tracked_objects.keys():
                self.kalman.predict(self.tracked_objects[key])
            # tenta associar os novos objetos com os kalmans
            # -----------------------------------------------
            # ------------- from pyimageresearch ------------
            # grab the set of object IDs and corresponding centroids
            object_IDs = list(self.tracked_objects.keys())
            H = np.array([[1., 0, 0, 0, 0, 0],[0., 1., 0, 0, 0, 0]])
            predictions = [H.dot(predicted_pos[1].x) for predicted_pos in self.tracked_objects.items()]

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
            D = dist.cdist(np.array(predictions), centroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
            
            try:
                rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
                cols = D.argmin(axis=1)[rows]
            except Exception as e:
                print('----------------------------')
                print("E: Nums = {} obj_b_tr = {}".format(nums,self.objects_being_tracked))
                print("Erro! D = {}".format(D))
                print('----------------------------')
                for key in self.tracked_objects.keys():
                    self.tracked_objects[key].missing_frames += 1
                    self.tracked_objects[key].missing = True
                return None
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			
            usedRows = set()
            usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
                if row in usedRows or col in usedCols:
                    continue
                if D[row][col] > self.max_radius:
                    continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				
                object_ID = object_IDs[row]
                self.tracked_objects[object_ID].z = centroids[col]
                self.kalman.update(self.tracked_objects[object_ID])
                self.tracked_objects[object_ID].missing_frames = 0
                self.tracked_objects[object_ID].missing = False

				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    object_ID = object_IDs[row]
                    self.tracked_objects[object_ID].missing_frames += 1
                    self.tracked_objects[object_ID].missing = True
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object

                    if self.tracked_objects[object_ID].missing_frames > self.max_missing_frames:
                        self.deregister(object_ID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(classes[col],scores[col],centroids[col])
            # -----------------------------------------------
            
            # os que foram associados recebem as novas posições com kalman.update
            # os objetos já registrados que não foram associados incrementam missing_frames em 1
            # objetos registrados que atingiram o max_missing_frames são desassociados
            # os objetos detectados nao associados sao registradosksys.
            
            # for (obj_id,tr_obj) in self.tracked_objects.items():
            #     print('id:{}, class={}, Xpos({},{}), Zpos({},{})'.format(obj_id,tr_obj.class_name,int(tr_obj.x[0]),int(tr_obj.x[1]),int(tr_obj.z[0]),int(tr_obj.z[1])))
            # print(D)
            
    
    def update(self, detected_objects): ## Vai se chamar update?
        self.dataAssociation(detected_objects)
        
        # Roda o update de todos os objetos detectados
