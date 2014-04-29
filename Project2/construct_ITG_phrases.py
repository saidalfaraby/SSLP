from collections import defaultdict


class Node (object):
    """docstring for Tree """
    def __init__(self, label=None):
		# super(Tree , self).__init__()
		self.label = label
		self.left = None
		self.right = None


class Combo(object):
	def __init__(self):
		self.combos = set()

	def update_combos(self, new_combos):
		new_combos = {' '.join(map(str, i)) for i in new_combos}
		self.combos.update(new_combos)


# string = ''
def printTree(T):
	def recur(T):
		if T.label is not None:
			return str(T.label)+' '
		else:
			string = ''
			string += recur(T.left)
			string += recur(T.right)
			return string
	return recur(T)


def getITGPermut(lb, ub):
	'''
	lb : lower bound of permuting integer
	ub : upper bound of permuting integer
	return set of itg-constrained string
	'''
	def recur(start, end):
		if end - start <= 1:
			return [Node(start)]
		else:
			all_t = []
			for k in xrange(start+1, end):
				for i in recur(start, k):
					for j in recur(k, end):
						for l, r in [(i, j), (j, i)]:
							t = Node()
							t.left = l
							t.right = r
							all_t.append(t)
			return all_t

	allString = set()
	allItg = recur(lb, ub)
	for t in allItg:
		allString.add(printTree(t))
	return allString


def permut_phrases(phrase_pairs, itg_permut):
	'''
	phrase_pairs is list of phrase_pair to be itg-constructed
	itg-permut is list of list of integer permutation
	we will permute the english side of pairs, assume right side of tuple
	return : list of tuple contain permuted join phrases
	'''
	all_permuted = []
	for itg in itg_permut:
		#allocate list
		right = ['' for i in range(len(phrase_pairs))]
		left = []
		itglist = itg.split()
		for i in range(len(itglist)):
			left.append(phrase_pairs[i][0])
			right[itglist[i]] = phrase_pairs[i][1]
		all_permuted.append((' '.join(left), ' '.joint(right)))
	return all_permuted


def devide_phrases_accto_length(all_phrase_pairs):
	'''
	return dictionary with length as key and list of phrase_pairs as value
	we will take max length between english and foreign phrase
	'''
	div_phrases = defaultdict(list)
	for p in all_phrase_pairs:
		div_phrases[max(len(p[0].split()), len(p[1].split()))].append(p)
	return div_phrases


def construct_itg_phrases(all_phrase_pairs):
	div_phrases = devide_phrases_accto_length(all_phrase_pairs)
	max_length = 4
	#find all combination of length that sum to 2 up to max_length
	length_combination = []
	for l in range(2, max_length+1):
		for m in range(2, l-1):
			pass


def find_combos(N):
	def subset_sum(numbers, target, cmb, partial=[], so_far=[]):
		s = sum(partial)
		# check if the partial sum is equals to target
		if s == target:
			so_far.append(partial)
			cmb.update_combos(so_far)
			# print "sum(%s)=%s" % (partial, target)
		if s >= target:
			return  # if we reach the number why bother to continue
		for i in range(len(numbers)):
			n = numbers[i]
			remaining = numbers[i+1:]
			subset_sum(remaining, target, cmb, partial + [n], so_far)
		#print [map(' '.join, i) for i in so_far]

	cmb = Combo()
	elements = []
	for i in xrange(1, N):
		for k in xrange(int(N/i)):
			elements.append(i)
	subset_sum(elements, N, cmb)
	return cmb


def test():
	def findAllSumTo(N):
		if N == 0:
			return [[]]
		else:
			choose = []
			for j in range(1, N+1):
				choose.append(j)
				N -= j
				for i in findAllSumTo(N):
					print i
					#choose.append(i.append(j))
			return choose

	t = findAllSumTo(4)
	print len(t)
	for i in t:
		print i


if __name__ == '__main__':
	# allString = getITGPermut(1,5)
	# print len(allString)
	# for s in allString:
	# 	print s
	cmb = find_combos(4)
	print cmb.combos
	# print cmb.combos
