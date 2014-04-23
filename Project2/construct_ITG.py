class Node (object):
	"""docstring for Tree """
	def __init__(self, label=None):
		# super(Tree , self).__init__()
		self.label = label
		self.left = None
		self.right = None


class ITG(object):
	"""docstring for ClassName"""
	def __init__(self):
		pass

	def recur(self, start, end):
		if end - start <=1:
			return [Node(end)]
		else :
			all_t = []
			for k in xrange(start+1,end):
				# t = Node()
				for i in self.recur(start,k):
					for j in self.recur(k,end):
						for l,r in [(i,j),(j,i)]:
							t = Node()
							t.left = l
							t.right= r
							all_t.append(t)
			return all_t

	def printTree(self, T):
		self.string  = ''
		def recur(T):
			if T.label!=None :
				self.string += str(T.label)+' '
			else :
				recur(T.left)
				recur(T.right)
		recur(T)
		return self.string

if __name__ == '__main__':
	itg = ITG()
	allItg = []
	allItg = itg.recur(0,4)
	allString = set()
	print len(allItg)
	for t in allItg:
		string = itg.printTree(t)
		allString.add(string)

	print len(allString)
	for s in allString:
		print s


		