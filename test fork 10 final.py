# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:50:08 2022

@author: rustt
"""
import random
import numpy as np 

from copy import deepcopy

dimension=2 
bounds = [(0, 1) for i in range(dimension)]

def Sphere(x):
    return sum(np.power(x, 2))

minf=0
maxf=1 

def custom_sample():
            
        
            #cas bounds =None 
            #faire même cas que la haut pour etre propre, parcourir tableau 
        if bounds is None:
                minf=0
                maxf=1
                
                return np.repeat(minf, repeats=dimension) \
               + np.random.uniform(low=0, high=1, size=dimension) *\
               np.repeat(maxf - minf, repeats=dimension)
            
        else:
                
                #peut prendre random bound dans array de bound defini précédemment
              #randompop=[]
                randomsamp=[]
              #for i in range(pop_size):
                for j in range(dimension):
                    #vérifier adéquat avec equation scholarpedia
                    randomsamp.append(random.uniform(bounds[j][0], bounds[j][1]))
                
        return randomsamp

population=10000

currentFitness=0 
        #var statut et co 
employedBees=[]
onlookerBees=[]

#array of array, ou 3 array
employedBeesTrial=[]
onlookerBeesTrial=[]

employedBeesProba=[]
onlookerBeesProba=[]

employedBeesFit=[]
onlookerBeesFit=[]

#GLOBALES
#variable Globale "objet" ABC


# # # # On met dans le def pour l'instant
#optimalityTracking=[]
#optimalSolution = []

# A PRENDRE AILLEURS PEUT ETRE 
max_trials=100
n_iter=100

#minf maxf globaux 


# à verif, sinon faire à la main, ou meme chose
#dans uatre fonc au dessus, voir 
def __reset_algorithm(optimalSolution,optimalityTracking):
    #on peut surement passer en paramètre, pas
    #besoin de fonction mais suit ABC

    # # # voir comment faire, sinon boucler dehors 
    optimalSolution= []
    optimalityTracking = []
    #return both 
    
def __update_optimality_tracking(optimalSolution,optimalityTracking):
    #on evalue la fonction. la on met sphere . evaluate, line 19
    #voir si on remplit via dim, ou si on fait une liste de liste 
    
    #rajouter liste en compréhension avec dim 
    print("OptimalSol", optimalSolution)
    oSol=np.array( [optimalSolution[0], optimalSolution[1] ])
    
    optimalityTracking.append(Sphere(oSol))
    return optimalityTracking
    
def __update_optimal_solution(optimalSolution):
    #min tout le tableau
   # newoptimalSolution=[]

    #FORMAT + max 
    newOptimalSolution=[0.0,0.0]
    #      indexes 
    
    #devrait s'appeler max
    minEmployeedFit= employedBeesFit.index(max(employedBeesFit))
    minOnlookerFit= onlookerBeesFit.index(max(onlookerBeesFit))
    
    #va prendre le min ? 
    #faut prendre max non ? 
    print("Employed fits", employedBeesFit[0:10])
    print("Onlooker fits", onlookerBeesFit[0:10])
    print("employedbeeslook 0" ,employedBees[0:10])
    print("Onlookerbeeslook 0" ,onlookerBees[0:10])
    print(" mins checks", minEmployeedFit, minOnlookerFit)
    if (employedBeesFit[minEmployeedFit] > onlookerBeesFit[minOnlookerFit]):
        #pb avec index 
        newOptimalSolution[0]= employedBees[minEmployeedFit][0]
        newOptimalSolution[1]= employedBees[minEmployeedFit][1]
    
    else:
        newOptimalSolution[0] = onlookerBees[minOnlookerFit][0]
        newOptimalSolution[1] = onlookerBees[minOnlookerFit][1]
        # # # A voir si ça marche ^^^^ 
        
        # # # 
        # pour être sur, on pourrait fouiller dans les 2, et prendre le min global, c'est propre 
        
        #pb, on voudrait l'index, pas le fit ? recup le min des deux, pas concatenation
        #faire la somme permet de trouver l'indice direct dans l'un des deux, et ensuite on cherche la bonne valeur
        #yes, good 

    
    # # # ou bien pb dans remplissage des tableaux , ou bien pb dans la copie/changement de optimalSolution
            
    print("Checkpoint", optimalSolution, newOptimalSolution, employedBees[minEmployeedFit], minEmployeedFit)
    print("employedbeeslook" ,employedBees)
    if len(optimalSolution)==0:
        optimalSolution=deepcopy(newOptimalSolution)
    else:
        #pareil, faire cas dim
        if ( Sphere( np.array( [ newOptimalSolution[0], newOptimalSolution[1] ]) < Sphere( np.array( [ optimalSolution[0], optimalSolution[1] ]) ))):
            optimalSolution=deepcopy(newOptimalSolution)
    
    #return versus void qui manipule
    return optimalSolution

def __employee_bees_phase():
    #explore max_trials de fois sur employeeBees
    for i in range(0,len(employedBees)):
        explore(i, max_trials)

def __calculate_probabilities():
    #pas besoin de map, on sum sur l'array 
    
    #somme des fitness des abeilles employées 
    #vérif bonne façon de sommer 
    sum_fitness = sum(employedBeesFit)
    
    #compute les proba de chaque employedBee
    # ce qui est définit par self getfit/max_fit 
    # array fit / max_fit 
    
    #remplir prob, utiliser get fit pour cas; 
    
    #np array vs array 
    #reference, passer ou non ? 
    for i in range(0,len(employedBees)):
        employedBeesProba[i]=get_fitness(i)/( sum_fitness + 0.000001) 
        #pb de division par zero, on rajoute un epsilon? 
        
        # # # 
        #epsilon a choisir ou bien faire un if si nul 
        
    #compute prob stockée qqpart? 
    
    #renvoyer sum ou 
    
    #no return sur deuxieme ligne du code, 
    #palle une méthode ? 
    
    #vérif. verif pas utilisation de copute prob ailleurs 

#Appartient à abc dans le medium 

best_food_sources=[]

print("LA LONGEUR", len(best_food_sources))

# # # 
#Rajout de bfs en paramètre pour que soit changé 

def __select_best_food_sources(best_food_sources):
    #check PASSE BIEN TABLEAU DEHORS DE LA FONCTION
    
    #array bfoodsources ? 
    #on filtre sur les proba des employedbees 
    
    #on recup l'indice viaemployedBeesProba, les proba on s'en fout, on veut les vecteurs pos associés 
    
    # on regarde les proba, et on met seulement dans la liste les bonnes employedBees 
    
    # # # 
    #est - ce que cette liste reste vide ? 
    for i in range(0,len(employedBees)):
        if employedBeesProba[i] > np.random.uniform(low=0, high=1) :
            best_food_sources.append(employedBees[i])
        
    #best_food_sources = [employedBees[i] for i in range(0, len(employedBees)) if (employedBeesProba[i] > np.random.uniform(low=0, high=1) ) ]
    #print("************************************* ********************************", best_food_sources)
    #while not best_food_sources:
    while ( len(best_food_sources)==0):
            for i in range(0,len(employedBees)):
                if employedBeesProba[i] > np.random.uniform(low=0, high=1) :
                    best_food_sources.append(employedBees[i])
                    
# # # WORKED BUT BAD DIM 
### change random choice nous même ? 
        

def __onlooker_bees_phase():
    #sur onlooker bees 
#map(lambda bee: bee.onlook(self.best_food_sources, self.max_trials),
#            self.onlokeer_bees)
    #onlook sur la liste des onlooker avec bfs et maxt
    for i in range(0, len(onlookerBees)):
        onlook(i, best_food_sources, max_trials)
        
def __scout_bees_phase():
    #reset all ? 
    #ou reset concatenation? même chose ? 
    #itérer sur les deux listes. doute 
    
    #on peut chercher le trial correspondant 
    #dans le tableau onlooker? 
    for i in range(0,len(onlookerBees)):
        trial=onlookerBeesTrial[i]
        reset_onlookerBee(i, trial,max_trials)
        
    for i in range(0,len(employedBees)):
        trial=employedBeesTrial[i]
        reset_employedBee(i, trial,max_trials)
        
#? 
#when is trial defined in reset? 
    
def evaluate_boundaries(pos):
    #boucler sur toutes les pos dehors de la fonc
        if (pos < minf).any() or (pos > maxf).any():
            pos[pos > maxf] = maxf
            pos[pos < minf] = minf
        return pos
    
def update_employedBee(i, pos, fitness):
        if fitness <= currentFitness:
            employedBees[i][0]=pos[0]
            employedBees[i][1]=pos[1]
            fitness = currentFitness
            print( employedBeesTrial[pos])
            employedBeesTrial[i] = 0
        else:
            employedBeesTrial[i] += 1
            
def update_onlookerBee(pos, fitness):
        if fitness <= currentFitness:
            pos = pos
            fitness = currentFitness
            onlookerBeesTrial[pos] = 0
        else:
            onlookerBeesTrial[pos] += 1
            
def reset_employedBee(i,trial, max_trials):
        if trial >= max_trials:
            __reset_employedBee(i)
            
def reset_onlookerBee(i,trial, max_trials):
        if trial >= max_trials:
            __reset_onlookerBee(i)

def __reset_employedBee(i):
        new = custom_sample()
        #itérer sur dimension quand marche. la juste deux premiers 
        
        #erreurs sur employedBees format. vecteur (a,b...,n) et la ai écrit successif ! 
        
        # # # 
        #Il faudra itérer sur dim de [i]:[1....n]
        
        employedBees[i][0] = new[0]
        employedBees[i][1]= new[1]
        #on pourra mettre un paramètre pour choisir la fonction. Sphere, Rast... 
        employedBeesFit[i] = Sphere(np.array( employedBees[i] ))
        #valeur initiale Trial peut être définie ailleurs 
        employedBeesTrial[i] = 0 
        employedBeesProba[i] = 0.0
        
def __reset_onlookerBee(i):
        new = custom_sample()
        #itérer 
        onlookerBees[i] = new[0]
        onlookerBees[i+1]= new[1]
        #on pourra mettre un paramètre pour choisir la fonction. Sphere, Rast... 
        
        #check bien la même mais devrait 
        onlookerBeesFit[i] = Sphere(np.array( employedBees[i]) )
        #valeur initiale Trial peut être définie ailleurs 
        onlookerBeesTrial[i] = 0 
        onlookerBeesProba[i] = 0.0
        
#meat and potatoes : les updates 

#employeeBee explore

#?changement de type ? employee->onlooker? 


#FONCTIONS EMPLOYEDBEES 

#paramètre de dimension à mettre ou a détecter dedans pour component, phi... 
def explore(i, max_trials):
        if employedBeesTrial[i] <= max_trials:
            #for in len, for in in range parametre dim, np.array 0..i....dim
            pos= np.array(employedBees[i])
            #on peut enlever redondance en dessous si on utilise pos
            
            #on va flatten. 
            #ou bien on choisit pos dimension d'un ind, pas 
            # individu i ou i+1 
            component = np.random.choice(np.array(employedBees[i]))
            phi = np.random.uniform(low=-1, high=1, size=len( np.array( employedBees[i] )))
            n_pos = pos + (pos - component) * phi
            n_pos = evaluate_boundaries(n_pos)
            
            #passage de la fonction ici
            #vérifier qu'on peut passer un np array, normalement fait pour ^
            n_fitness = Sphere(n_pos)
            update_employedBee(i,n_pos, n_fitness)

def get_fitness(i):
   return 1 / (1 + employedBeesFit[i]) if employedBeesFit[i] >= 0 else 1 + np.abs(employedBeesFit[i])

def compute_prob(i, max_fitness):
   employedBeesProba[i] = get_fitness(i) / max_fitness
   
#FONCTIONS ONLOOKERBEES

# # # 
#pb avec best food sources qui est ini vide ! 

def onlook(i, best_food_sources, max_trials):
        #print(best_food_sources)
        randIndex=random.randrange(0, len(best_food_sources),1)
        candidate = best_food_sources[randIndex]
        # # # on va tirer un index random, ça devrait fonctionner 
        #! ça marche 
        
        # # # bien sur on passera les fonctions 
        
        candidateFitness= Sphere(candidate)
        # # # candidate.pos et fit ???? existe pas. on peut les recup et  calculer direct ici 
        __exploit(i,candidate, candidateFitness, max_trials)

def __exploit(i, candidate, fitness, max_trials):
        if onlookerBeesTrial[i] <= max_trials:
            component = np.random.choice(candidate)
            phi = np.random.uniform(low=-1, high=1, size=len(candidate))
            n_pos = candidate + (candidate - component) * phi
            n_pos = evaluate_boundaries(n_pos)
            #use f instead of sphere, and pass f in arg way to the top
            n_fitness = Sphere(n_pos)

            if n_fitness <= fitness:
                #pareil, on peut passer paramètre dim chaque fois que necessaire, pour itérer 
                
                #attention à copie vs reference . on veut reference 
                #on peut faire une boucle pour update les originaux 
                dim=2 
                for i in range(0,dim):
                    onlookerBees[i]=n_pos[i]
                    
                #pos = n_pos
                onlookerBeesFit[i] = n_fitness
                onlookerBeesTrial[i] = 0
            else:
                onlookerBeesTrial[i] += 1
        

def __initialize_all():
    for i in range(0,population//2):
            #valeur
            employedBees.append(custom_sample())
            #nombre d'itérations
            employedBeesTrial.append(0)
            #Probabilité initiale 
            employedBeesProba.append(0.0)
            
            employedBeesFit.append(0.0)
            
            onlookerBees.append(custom_sample())
            #nombre d'itérations
            onlookerBeesTrial.append(0)
            #Probabilité initiale 
            onlookerBeesProba.append(0.0)
            
            onlookerBeesFit.append(0.0)
            
    #quand boucle finie, on met les valeurs initiales de proba, fitness !
    for i in range(0,population//2):
        #Sphere ou getfitness ? look
        employedBeesFit[i]= Sphere(employedBees[i])
    
    max_fitness= max(employedBeesFit)
    #check pas zero
    
    #check bonnes valeurs 
    for i in range(0,population//2):
        employedBeesProba[i]= employedBeesFit[i]/max_fitness
            

            
print(employedBees[0:2])


def optimize():
    #on pourrait stocker les var globales la dedans 
    optimalSolution=[]
    optimalityTracking=[]
    
    #voir si on fait pas des copies dans le vent
    
    #appeler les fonctions dans l'ordre et compagnie 
  #  __reset_algorithm(optimalSolution, optimalityTracking)
    #initiatisation en haut 
    __initialize_all()
    for i in range(n_iter):
        __employee_bees_phase()
        optimalSolution= __update_optimal_solution(optimalSolution)
        
        __calculate_probabilities()
        __select_best_food_sources(best_food_sources)
        
        __onlooker_bees_phase()
        __scout_bees_phase()
        
        print(optimalSolution)
        
        __update_optimal_solution(optimalSolution)
        optimalityTracking = __update_optimality_tracking(optimalSolution,optimalityTracking)
        print("iter: {} = cost: {}"
                  .format(i, "%04.03e" % Sphere(optimalSolution)))
        
optimize()