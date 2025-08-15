def determineMetaclass(bases, explicit_mc=None):

    metaclass_hierarchy = []

    metaclass = exp_mc or bases[0]

    for base in bases[1:]:

        metaclass_hierarchy.append(base)

        metaclass = metaclass.mro()[-2]


    return metaclass