import mlp
class trainedmlp():
	def __init__(self,net,testinput,testtarget,trainmeanerr,validmeanerr,numnodes):
		self.net=net
		self.testinput=testinput
		self.testtarget=testtarget
		self.trainmeanerr=trainmeanerr
		self.validmeanerr=validmeanerr
		self.numnodes=numnodes
	def test(self):
		return self.net.confmat(self.testinput,self.testtarget)