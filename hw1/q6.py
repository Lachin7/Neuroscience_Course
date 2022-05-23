
# 2: We can see that by increasing $M_{EE}$,
import numpy as np

Mee = 1.25
Mei = -1
Mie = 1
Mii = 0
Ye = -10
Yi = 10
Te = 0.01
Ti = 0.05
eigenvalue_1 = 1/2 * (((Mee-1)/Te)+((Mii-1)/Ti)+np.sqrt(np.power(((Mee-1)/Te)-((Mii-1)/Ti), 2)+(4*Mei*Mie)/(Ti * Te)))
eigenvalue_2 = 1/2 * (((Mee-1)/Te)+((Mii-1)/Ti)-np.sqrt(np.power(((Mee-1)/Te)-((Mii-1)/Ti), 2)+(4*Mei*Mie)/(Ti * Te)))
print(eigenvalue_1)
print(eigenvalue_2)