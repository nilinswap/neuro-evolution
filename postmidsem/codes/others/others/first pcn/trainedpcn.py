import pcn
class trainedpcn():
	def __init__(self,net,testinput,testtarget,validmeanerr,eta,niterations):
		self.net=net
		self.testinput=testinput
		self.testtarget=testtarget
		self.validmeanerr=validmeanerr
		self.eta=eta
		self.niterations=niterations
	def test(self):
		return self.net.confmat(self.testinput,self.testtarget)