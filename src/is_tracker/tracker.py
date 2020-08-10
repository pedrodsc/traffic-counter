# TODO
# Trocar a posicão dos objetos de absoluta pra [0,1]
# como a saída do YOLO
import numpy as np
from numpy.linalg import inv
from collections import OrderedDict
from scipy.spatial import distance as dist
from options_pb2 import KalmanParams, TrackerOptions

class TrackedObject(object):
    def __init__(self,object_ID,class_id,score,x_pos,y_pos,bbox):
        self.class_id = int(class_id)
        self.object_ID = object_ID
        self.score = score
        self.missing_frames = 0
        self.missing = False
        self.path = []
        # Covariance matrix - Centroid
        self.P = np.eye(6)
        self.x = np.array([x_pos,y_pos,0,0,0,0],dtype=np.float)
        self.z = self.x[0:2]
        # Covariance matrix - BBox
        self.Pb1 = np.eye(6)
        self.xb1 = np.array([bbox[0],bbox[1],0,0,0,0],dtype=np.float)
        self.zb1 = self.xb1[0:2]
        self.Pb2 = np.eye(6)
        self.xb2 = np.array([bbox[2],bbox[3],0,0,0,0],dtype=np.float)
        self.zb2 = self.xb2[0:2]
        self.bbox = bbox



class Kalman(object):
  def __init__(self,KalmanParams):
    
    self.R = np.eye(2)*KalmanParams.R # Measure noise
    self.Q = np.eye(6)*KalmanParams.Q # Process noise
    self.dt = KalmanParams.dt # Ainda acho que será útil ter isso aqui
    dt = self.dt # Feio, eu sei, mas não vou ficar repetindo self toda hora
    # Matrix to extract Cx and Cy from vector x
    self.H = np.array([[1., 0, 0, 0, 0, 0],[0., 1., 0, 0, 0, 0]])
    # Model of the process
    self.A = np.array( [[1., 0, dt,  0, (dt**2)/2, 0],     # Cx = Cx + Vx*dt + 0.5*Ax*dt**2
                        [0,  1,  0, dt, 0,     (dt**2)/2], # Cy = Cy + Vy*dt + 0.5*Ay*dt**2
                        [0,  0,  1,  0, dt,    0],         # Vx = Vx + Ax*dt
                        [0,  0,  0,  1, 0,     dt],        # Vy = Vy + Ay*dt
                        [0,  0,  0,  0, 1,     0],         # Ax = Ax
                        [0,  0,  0,  0, 0,     1]])        # Ay = Ay

  def predict(self,tracked_object):
    
    tracked_object.x = self.A.dot(tracked_object.x) # (C.5)
    tracked_object.P = self.A.dot(tracked_object.P).dot(self.A.T) + self.Q # (C.6)

    tracked_object.xb1 = self.A.dot(tracked_object.xb1) # (C.5)
    tracked_object.Pb1 = self.A.dot(tracked_object.Pb1).dot(self.A.T) + self.Q # (C.6)

    tracked_object.xb2 = self.A.dot(tracked_object.xb2) # (C.5)
    tracked_object.Pb2 = self.A.dot(tracked_object.Pb2).dot(self.A.T) + self.Q # (C.6)

    tracked_object.bbox[0] = tracked_object.xb1[0]
    tracked_object.bbox[1] = tracked_object.xb1[1]
    tracked_object.bbox[2] = tracked_object.xb2[0]
    tracked_object.bbox[3] = tracked_object.xb2[1]

  def update(self,tracked_object):
    # Centroid
    S = self.H.dot(tracked_object.P).dot(self.H.T) + self.R # (C.9)
    K = tracked_object.P.dot(self.H.T).dot(inv(S)) # (C.9)
    y = tracked_object.z - self.H.dot(tracked_object.x) # (C.10)
    tracked_object.x = tracked_object.x + K.dot(y) # (C.10)
    tracked_object.P = tracked_object.P - K.dot(self.H).dot(tracked_object.P) # (C.11)

    # Corner 1
    S = self.H.dot(tracked_object.Pb1).dot(self.H.T) + self.R # (C.9)
    K = tracked_object.Pb1.dot(self.H.T).dot(inv(S)) # (C.9)
    y = tracked_object.zb1 - self.H.dot(tracked_object.xb1) # (C.10)
    tracked_object.xb1 = tracked_object.xb1 + K.dot(y) # (C.10)
    tracked_object.Pb1 = tracked_object.Pb1 - K.dot(self.H).dot(tracked_object.Pb1) # (C.11)

    # Corner 2
    S = self.H.dot(tracked_object.Pb2).dot(self.H.T) + self.R # (C.9)
    K = tracked_object.Pb2.dot(self.H.T).dot(inv(S)) # (C.9)
    y = tracked_object.zb2 - self.H.dot(tracked_object.xb2) # (C.10)
    tracked_object.xb2 = tracked_object.xb2 + K.dot(y) # (C.10)
    tracked_object.Pb2 = tracked_object.Pb2 - K.dot(self.H).dot(tracked_object.Pb2) # (C.11)

    tracked_object.bbox[0] = tracked_object.xb1[0]
    tracked_object.bbox[1] = tracked_object.xb1[1]
    tracked_object.bbox[2] = tracked_object.xb2[0]
    tracked_object.bbox[3] = tracked_object.xb2[1]

class Tracker(object):
    def __init__(self, img_shape, TrackerOptions):
        # Lista, vetor ou dicionário de objetos sendo rastreados
        self.tracked_objects = OrderedDict()
        # Numero de objetos já rastreados
        self.next_obj_ID = 1
        # Número de objetos sendo rastreados
        self.objects_being_tracked = 0
        # Hiperparâmetros: Limite de frames para remover um objeto, incertezas*
        self.max_missing_frames = TrackerOptions.max_missing_frames
        # Não sei se as incertezas serão para cada objeto ou global
        
        # Distancia maxima pra tentar associar objetos
        self.max_radius = TrackerOptions.max_radius
        # This matrix times bbox(0 to 1) gives the centroid of the bbox
        (w,h) = img_shape
        self.centroid_matrix = np.array([[w/2,0,w/2,0],[0,h/2,0,h/2]],dtype=np.float)
        self.bbox_matrix = np.array([[w,0,0,0],[0,h,0,0],[0,0,w,0],[0,0,0,h]],dtype=np.float)
        self.kalman = Kalman(TrackerOptions.KalmanParams)
    
    def register(self, class_id, score, centroid, bbox):
        x_pos = centroid[0]
        y_pos = centroid[1]
        self.tracked_objects[self.next_obj_ID] = TrackedObject(self.next_obj_ID,class_id,score,x_pos,y_pos,bbox)
        self.next_obj_ID += 1
        self.objects_being_tracked += 1

    def deregister(self, object_ID):
        self.tracked_objects.pop(object_ID)
        self.objects_being_tracked -= 1
    
    def untrack_all(self):
        self.tracked_objects.clear()
        self.objects_being_tracked = 0
        self.next_obj_ID = 1

    def dataAssociation(self, detected_objects):
        # Associa os novos dados aos objetos antigos
        # Adiciona novos objetos

        # Parse detected_objects
        boxes, scores, classes, nums = detected_objects

        # centroids in x,y format
        centroids = np.empty((nums,2))      
        bboxes = np.empty((nums,4))
        for i in range(nums):
            box = boxes[i]
            #                               |bx1|
            #  |Cx| = |w/2  0  w/2 0  |  *  |by1|
            #  |Cy|   | 0  h/2  0 h/2 |     |bx2|
            #                               |by2|
            centroids[i] = self.centroid_matrix.dot(box)
            bboxes[i] = self.bbox_matrix.dot(box)
        if self.objects_being_tracked == 0:
            for i in range(nums):
                self.register(classes[i],scores[i],centroids[i],bboxes[i])
        else:
            # Predição dos objetos
            for key in self.tracked_objects.keys():
                self.kalman.predict(self.tracked_objects[key])
            # Tenta associar os novos objetos com as predições dos kalmans
            # -----------------------------------------------
            # ------------- from pyimageresearch ------------
            # grab the set of object IDs and corresponding centroids
            object_IDs = list(self.tracked_objects.keys())
            H = np.array([[1., 0, 0, 0, 0, 0],[0., 1., 0, 0, 0, 0]])
            predictions = [H.dot(predicted_pos[1].x) for predicted_pos in self.tracked_objects.items()]
            # Calcula a distância entre cada par de centroides
            # e as predições do frame anterior
            D = dist.cdist(np.array(predictions), centroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list

            # Ordena a matriz pelas linhas e em seguida pelas colunas
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
                self.tracked_objects[object_ID].class_id = classes[col]
                self.tracked_objects[object_ID].z = centroids[col]
                self.tracked_objects[object_ID].zb1 = bboxes[col][0:2]
                self.tracked_objects[object_ID].zb2 = bboxes[col][2:4]
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
                    self.register(classes[col],scores[col],centroids[col],bboxes[col])
            # -----------------------------------------------

            # os que foram associados recebem as novas posições com kalman.update
            # os objetos já registrados que não foram associados incrementam missing_frames em 1
            # objetos registrados que atingiram o max_missing_frames são desassociados
            # os objetos detectados nao associados sao registradosksys.

            # create paths
            for (obj_id,obj) in self.tracked_objects.items():
                obj.path.append(obj.x[0:2])

    def find(self, class_id):
        obj_id_list = []
        for key in self.tracked_objects.keys():
            if self.tracked_objects[key].class_id == class_id:
                obj_id_list.append(self.tracked_objects[key].object_ID)
        return obj_id_list

    def update(self, detected_objects):
        self.dataAssociation(detected_objects)

        # Roda o update de todos os objetos detectados