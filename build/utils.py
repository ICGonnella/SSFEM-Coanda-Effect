
def separate(vect, dof_u, dof_p):
    return [vect[:dof_u][::2], vect[:dof_u][1::2], vect[-dof_p:]]