# Temporal-Difference Learning Methods in Mountain Car v0 Open AI: Q-learning, SARSA and Expected SARSA

TD learning is a combination of Monte-Carlo ideas and dynamic programming ideas (Sutton). These methods only need next step to get the next reward and the estimate value function. The simplest equation for a TD method is
![TD-simplest-eq](../images_theory/TD-simplest_2.png)

where the quantity in brackets is an error between an estimated value and the best estimated.

They have the possibility of learn directly with **no model**, and they **boostrap**, it means that they learn a guess form a guess. But all use the experience to learn.
Other important feature is that they are implemented **on line**, because they only need wait one time step. Unlike Monte-Carlo methods that they need to go to the end to pick up the rewards.

These methods can be used to prediction task or evaluation policies, where they assess value function for a given policy. Even for control task, to finding optimal policy.


## Assessing Q Learning, SARSA and expected SARSA algoritmhms

We are going to work with Open AI environment, MountainCar v0. This environment shows a car in the valley between two mountains, and we try to get the right place where is the goal. There are discrete actions number, 0 to left accelerate, 1 no accelerate and 2 right accelerate, and a discrete state space, 0 represents position with continuous values [-1.2, +0.6] and 1 represents car velocity between [-0.07, +0.07]. More info [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py)


## Hyperparams and Table

    - **Table size**: we are going to discretize states and its correspondant actions. Recall states going to values -1.2 to +0.6, so we create 20x20, 40x40 and 80x80 tables, to save states values, eg., in table of 20x20 we save states in rows (in every cell we have a range of 0.09 size , starting from -1.2 to -1.11 and so on) and actions in colums. See figure below. 
    In this exercise we try to understand and clarify how the methods behavior in a tabular space. Likely in a real environment won't work and we would need a bigger table, doing unpractical these algorithms, but for understanding purposes is very clear.
    - Table initialitation random values. We random between -2 and 0. See below more explanation and how affects us environment rewards.

    - **Episodes**, we are going to exec up to 25.000
    - **epsilon**, range from (0,1], to control exploration against explotation, basic principle in RL. When epsilon is close to 1, algorithm explores most of time and it lasts long time to converge. We use a epsilon decay rate to balance towards explotation while execution.
    - **epsilon decay rate** to manage epsilon towards exploration or explotation.
    - **Learning Rate** or **alpha** or **step size** according [Sutton, eq 2.4] minimizes difference between goal and estimated values. Values are in range (0,1] and when it is close to 1, last reward gets the main weights. We implement it as a constant in this work but a future improvement could be implementing a decay rate, similar we made with epsilon.
    - **Discount rate** or **gamma** in range (0,1] which indicates importance of future values. When it is close to 1, it means we give importance recent values, and old values are less important for us.

    More features:

    As it is a closed environment, we import other values, such as 
    - initial position, always between [-0.6, -0.4]
    - rewards, 0 when we reach goal in position 0.5, -1 in all situations.

    It is important working with rewards features. In this environment, always we get a negative total reward. Table initialization will be random values between -2 and 0. We are going to exec values negatives and positives in order to find put how affect the results.


## Technical params

- Mac OS Big Sur 11.3
- Python 3.8
- gym 0.18

Other libraries can be found in requirements.txt



## Q Learning
It is a off policy method saving Q(s,a) values in our table. The Q Learning equation

![Ec-QLearning](../images_theory/Q-learning.png)

it gets Q(s,a) maximizing actions in next state. The backup diagram is

![backu-QLearning](../images_theory/QLearning_backup.png)

Q Learning approaches to optimal policy independent behavior policy, that's why it is a off-policy method. However policy determines state-action pair to visit and update.


### First Analysis to understand algorithmic behavior

In a first moment, we are going to try our algorithms and evaluate how they work with differents values.

- Episodes: 5000
- epsilon: 1
- epsilon decay value = epsilon / (episodes - 1), so algorithm exploit after the first half
- Table size: 40 x 40 x 3 (last one are actions)
- Learning rate or alpha: 0.1
- Discount rate: 0.95
- Q table random intialize between -2 and 0

![1-QLearning](../images_theory/QLearning-01-095-5000.png)

The Figure above shows us up to episode 3000, Q-Learning explores in the state space, after that it finds the goal. Then, we can see how the actions took place in episode 4000. 


![table-QLearning](../images_theory/QLearning_table1.png)


What we are seeing is our Q Learning table where we save actions value. Recall the table has a 40 x 40 size, where each cell represents the state space discretize and its correspondt action. So, row 40 and close, shows near position to goal (0.6) and column 40 the largest velocity got.
Cell in green shows the maximun result.
Actions are:
- 0 left accelerate 
- 1 do nothing
- 2 right accelerate

In the figure for Action 0, first graph, we can see that there is a green cluster in the botton rows for action 0, aprox. from 10 column to 25. There are left far positions where car is accelerating at maximun to get impulse and go to the right to reach the goal.
For action 2, acceleration to right, the cluster is the opposite of the previous, where the car is accelerating to right in positions close to goal.


### Analysis

- Episodes: 25.000
- epsilon: 1
- epsilon decay value = epsilon / (episodes - 1), so algorithm exploit after the first half
- Table size: 40 x 40 x 3 (last one are actions)
- Learning rate or alpha: 0.1
- Discount rate: 0.95
- Q table random intialize between -2 and 0


![2-QLearning](../images_theory/QLearning-01-095-25000.png)

From 10.000 episode, algorithm starts to find the goal. Similar first analysis, our epsilon and epsilon decay hyperparams, make the algorithm explore for the first half episodes. So we can see the importance of epsilon finding the solution.
In the second half, it seems our results incresing steadily.

![table2-QLearning](../images_theory/QLearning_table2.png)

Remember that action 1 is not to accelerate, that is, to leave the car loose. Action 0 accelerate to the left and action 2 accelerate to the right. It is seen that in the positions close to the objective (rows 40 and adjacent) the chosen action has been to accelerate to the right to get closer to the objective. While in positions near the opposite end of the goal, choose action 0, or accelerate to the left to gain momentum. Action 1 is also taken close to the goal, where it seems that the car already has a lot of momentum to get there. This may be due to past episodes where the car was swaying randomly and was recorded in our Q table

### Next analysis, all table size 40 x 40, 25.000 episodes and epsilon and epsilon decay similar


- Learning rate or alpha: 0.20
- discount rate: 0.95

![3-QLearning](../images_theory/QLearning-020-095.png)

As we go to increasing alpha, we are giving more importance last results, according our Q Learning formula, as we can check. So, old values have less importance and we could expect better convergence. But the other hyperparamns could affect as well.

- Learning rate or alpha: 0.20
- discount rate: 0.80

![4-QLearning](../images_theory/QLearning-02-080.png)


- Learning rate or alpha: 0.20
- discount rate: 0.5

![5-QLearning](../images_theory/QLearning-02-050.png)



- Learning rate or alpha: 0.5
- discount rate: 0.95

![6-QLearning](../images_theory/QLearning-05-095png.png)



- Learning rate or alpha: 0.5
- discount rate: 0.8

![7-QLearning](../images_theory/QLearning-05-08.png)


- Learning rate or alpha: 0.5
- discount rate: 0.5

![8-QLearning](../images_theory/QLearning-05-05.png)


- Learning rate or alpha: 1
- discount rate: 0.95

![8-QLearning](../images_theory/QLearning-1-095.png)

As we can see, when we've changed hyperparams, our algorithm has solved in worst convergence.

Now, we see in the next graphs, changing table size and others values


### Table 20 x 20

- Learning rate or alpha: 0.1
- discount rate: 0.95

![9-QLearning](../images_theory/QLearning-01-095-20x20.png)


### Table 80 x 80

- Learning rate or alpha: 0.1
- discount rate: 0.95

![10-QLearning](../images_theory/QLearning-01-095-80x80.png)

![11-QLearning](../images_theory/QLearning_table_80x80.png)


We can see in rows close to 80, the goal, the most choose action is 2, accelerate to right. 



### Epsilon 

In the last analysis, our decay rate was 0.00004, and recall that means Q Learning was exploring during long time until it could got good results. In next analysis, we try with differents values

- Table 40 x 40
- learning rate: 0.1
- discount rate: 0.95
- epsilon decay: 0.00016


![12-QLearning](../images_theory/QLearning-01-095-00016.png)

The picture shows us the importance of epsilon, where Q-Learning finds good results from episode 6000 aprox. instead more than 10.000 episodes before seen. So, even gets better results in global. Epsilon controls exploration versus explotation, one of the biggest issues in RL.



- Table 40 x 40
- learning rate: 0.1
- discount rate: 0.95
- epsilon decay: 0.004


![12-QLearning](../images_theory/QLearning-01-095-004.png)




# Changing manually method in execution to automatically

Once we can check Q Learning algorithm, theory and how hyperparams affect results in MountainCar Open Ai environment, in the next analysis, we modify our algorithm to find automatically the best results in differents hyperparams for Q Learning, SARSA and Expected SARSA, which are all of them, TD methods and they help us to well understand in order to use in all kind of environments, and try to choose the best for any kind of problem.


But before doing it, next we explain SARSA and Expected SARSA methods

## SARSA
SARSA is an on-policy method, which means it follows a previous fixed policy. SARSA learns a state-action function, where it has to estimate Q(s,a) for a policy PI and for all states and actions. The name of SARSA stands for State, Action, Reward, State next, Action next which are the steps SARSA has to take.
The eq. is:

![13-SARSA](../images_theory/SARSA-eq.png)

and its backup diagram follows the diagram:

![14-SARSA](../images_theory/SARSA-backup.png)





## Expected SARSA
This algorithm behaviors as Q Learning method, but instead taking the action that maximize the result, it takes the probability of each action under the policy determined. The eq. is:

![15-expectedSARSA](../images_theory/expected-SARSA-eq.png)
and its diagram

![16-expectedSARSA](../images_theory/expected-SARSA-backup.png)

This algorithm can be used in on-policy or off-policy, where in the last case it can use a different policy. For example, if the policy is e-greedy and the behavior policy es more exploratory, expected SARSA behaviors Q Learning.





---

# In Spanish


## Q Learning algorithm


En este ejercicio vamos a trabajar el entorno MountainCar-V0 de OpenAi con el algoritmo Qlearning.

El entorno de trabajo se ha realizado con las siguientes características mas relevantes 
Mac Big Sur 11.3
python 3.8.6
gym 0.18.0
El resto de librerías se encuentran en el fichero requeriments.txt

El código se encuentra en el fichero QLearning_multiple.py

El entorno de OpenAI (https://gym.openai.com/envs/MountainCar-v0/) muestra un auto posicionado al inicio en un valle y tiene que alcanzar la meta que se encuentra en la parte derecha de una colina. 
El espacio de acciones es discreto, con solo 3 acciones, 0 acelerar izquierda, 2 acelerar derecha y 1 no acelerar.
El espacio de estados es discreto también con solo 2 estados, 0 representa la posición con rangos min y max [-1.2 ,  +0.6] y 1 representa la velocidad con rangos min y max [-0.07 , +0.07]. Si bien, solo hay 2 estados, los valores que contienen son continuos por lo que podremos considerar que es un entorno mixto continuo-discreto. 
Las características del entorno desarrollado las encontramos aquí: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py



El algoritmo QLearning es un método tabular que guarda los valores Q(s,a) en una tabla, por lo que debemos discretizar todos los valores que puede tomar el entorno.
Recordando la ecuación de Qlearning

![Ec-QLearning](qtable_charts/Q-learning.png)

Otra forma de ver la ecuación está en 6.8 del libro de Sutton:

![Ec-QLearning2](qtable_charts/QLearning-english.svg)


A continuación analizaremos los diferentes hiperparámetros del algoritmo para ver como se comportan los resultados

### Hiperparámetros

•	Tamano de la tabla

En el caso de los estados, vamos a analizar los resultados del algoritmo con tablas de diferentes tamaños, 20x20, 40x40, 80x80 para analizar el comportamiento del algoritmo en función del tamaño de los rangos que separamos los valores continuos. Es decir, en una tabla de 20x20, crearemos 20 rangos que separaran los valores de [-1.2 , +0.6] de manera proporcional para la variable posición y [-0.07 , +0.07] para la variable velocidad.
De esta manera discretizamos el espacio de estados.


•	Episodios, trabajaremos con valores desde 5.000 hasta 25.000 

•	Epsilon [1, 0] o variable que sirve para controlar la exploración frente a la explotación del algoritmo, analizaremos valores desde 1 hasta 0.1 y analizaremos los resultados. Cuando el valor esta cercano a 1, al algoritmo explora continuamente y tarda mucho en converger a buenos resultados. Iremos viendo como se comporta nuestro algoritmo.

•	Tasa de decaimiento de épsilon. Ajustaremos el valor para analizar si afecta a nuestros resultados

•	Tasa de aprendizaje (learning rate) según la ecuación 2.4 del libro de referencia [Sutton] es la variable StepSize que reduce la diferencia entre el valor objetivo y el valor estimado. El valor se encuentra entre (0,1] y si esta muy cercano a 1 entonces 1 – alfa = 0 por lo que  dara todo el peso a la ultima recompensa. Cuanto alfa este mas cercano a 0 y la ecuación 1-alfa Aprx a 1, otorgara mas peso a las recompensas pasadas. En nuestro algoritmo lo vamos a dejar constante en cada ejecución aunque haremos ejecuciones con diferentes valores. Una posible mejora es aplicar decaimiento según se van ejecutando episodios dentro de la misma ejecución.

•	Tasa de descuento (discount rate) [1 - 0], o variable que indica la importancia de valores futuros. Analizaremos diferentes valores

Al ser un entorno con valores predeterminados por OpenAI, nos devuelve automáticamente los valores de

•	El entorno también nos proporciona el estado inicial del carro en cada episodio, siempre en la posición entre [-0.6 y -0.4]

•	Recompensas: cada movimiento que no alcanza el objetivo tiene una recompensa de -1 y 0 solo cuando alcanzamos el objetivo. Esto hace que la suma de recompensas, aun en el mejor de los casos, sea negativa.


Ms adelante, nuestra intención es trabajar con diferentes valores de recompensas para analizar los resultados.


•	Inicialización de la tabla Qlearning, en donde vamos a crearla de manera aleatoria con valores entre -2 y 0. Mas adelante analizaremos como la inicialización de la tabla Q puede hacer que el algoritmo se comporte mejor. Valores a 0 o positivos.

### Primer análisis

•	Episodios: 5000
•	Épsilon: 1
•	Épsilon_decay_value = epsilon / (episodios - 1)
•	Tamano de la tabla: 40x40x3 (3 son las acciones)
•	Learning_rate = 0.1
•	Discount_rate= 0.95
•	Q tabla inicializada aleatoriamente con valores entre -2 y 0


![QLearning_1](qtable_charts/Figure_5000-01-095.png)
La ejecución de QLearning nos muestra que hasta el episodio 3400 aprox. El algoritmo no encuentra soluciones y se dedica a explorar (recordar que el entorno se inicializa si el carro encuentra el objetivo o han pasado 200 episodios sin alcanzarlo). Desde este episodio hasta el 4500 aprox el resultado se alcanza porque ha ido aprendiendo pero hay una caída al final que deberemos ir analizando. 



En la siguiente imagen vamos a ver como se ha distribuido las acciones en nuestra tabla Q para el episodio 4000.

![QLearning_1_1](qtable_charts/4000.png)

Recordemos que nuestra tabla Q que guarda nuestros valores, tiene un tamaño de 40 x 40. Cada celda representa el espacio de estados que lo hemos discretizado. Por lo tanto, la fila 40 del grafico muestra la posición mas cercana al objetivo (0.6) y la columna 40 representa la mayor velocidad tomada (0.07)

Las acciones son 

    0: acelerar a la izquierda
    1: no acelerar
    2: acelerar a la derecha

Los puntos de color verde representan el valor máximo de la tabla.

Podemos ver que hay un cluster de verdes en la acción 0, para las filas inferiores, desde la columna 10 hasta la 25 aproximadamente. Esto nos indica que son las posiciones alejadas a la meta en donde el carro esta acelerando al máximo para subir la colina de la izquierda y poder tomar fuerza para luego subir a la derecha y alcanzar la meta.
En la grafica de la acción 2 (acelerar a la derecha) el cluster encontrado es el inverso del anterior, en donde parece que el carro acelera a la derecha en las posiciones cercanas a la meta.


### Segundo análisis (25000 episodios)

Todas las variables son las mismas que en el primer análisis, pero vamos a ejecutar 25000 episodios para analizar el resultado.

![QLearning_2](qtable_charts/Figure_25000-01-095.png)

El algoritmo ha empezado a converger a partir de los 10000 episodios. Se ve una línea creciente de recompensas (o decreciente en el resultado) llegando a su optimo sobre los 23000 episodios.

Veamos la tabla Q en el episodio 23000 para analizar su resultado.

![QLearning_2_1](qtable_charts/23000.png)

Recordemos que la acción 1 es no acelerar, es decir, dejar el carro suelto. La acción 0 acelerar a la izquierda y la 2 acelerar a la derecha. Se ve que en las posiciones cercanas al objetivo (filas 40 y adyacentes) la acción elegida ha sido acelerar a la derecha para acercarse al objetivo. Mientras que en las posiciones cercanas al extremo opuesto de la meta, se elige la acción 0, o acelerar a la izquierda para tomar impulso. La acción 1 se toma también cercana a la meta, en donde parece que el carro ya tiene mucho impulso para llegar. Esto puede ser debido a los episodios pasados en donde el carro se balanceaba de manera aleatoria y quedaba registrada en nuestra tabla Q

### Vamos a ejecutar varias pruebas con todos los valores similares a los anteriores ejercicios (tabla tamaño 40 x 40, 25000 episodios, épsilon decay similar) y cambiamos Learning rate y Discount rate



    Learning rate : 0.20 y Discount rate : 0.95

![QLearning_3](qtable_charts/Figure_25000-02-095.png)

Recordemos que la tasa de aprendizaje (learning rate) si esta cercana a 0, esta dando mas importancia a los valores antiguos. 


    Learning rate : 0.20 y Discount rate : 0.80

 ![QLearning_4](qtable_charts/Figure_25000-02-08.png)   

 Si lo comparamos con la ejecución anterior (Learning rate : 0.2, Discount rate : 0.95) vemos que empieza a verse los primeros resultados a partir del episodio 10000 de manera similar, aunque parece que los resultados conseguidos no son tan buenos, mejor ejecución sobre los -110 puntos, frente a los -95 aprox de la anterior.
Dsicount rate de 0.80 indica que estamos dando mas importancia a los valores pasados según vamos decreciendo la variable.


    Learning rate : 0.20 y Discount rate : 0.50

 ![QLearning_4](qtable_charts/Figure_25000-02-05.png)   


 Obtenemos resultados peores en las recompensas


     Learning rate : 0.50 y Discount rate : 0.95

 ![QLearning_4](qtable_charts/Figure_25000-05-095.png)  


      Learning rate : 0.50 y Discount rate : 0.80

 ![QLearning_4](qtable_charts/Figure_25000-05-08.png)  

      Learning rate : 0.50 y Discount rate : 0.50

 ![QLearning_4](qtable_charts/Figure_25000-05-05.png)  


      Learning rate : 1 y Discount rate : 0.95

 ![QLearning_4](qtable_charts/Figure_25000-1-095.png)  


      Learning rate : 1 y Discount rate : 0.5

 ![QLearning_4](qtable_charts/Figure_25000-1-05.png)  


 Segun hemos ido cambiando los parametros, los resultados han sido peores. 


 Veamos como afecta el tamano de la tabla. Hasta ahora hemos realizado pruebas con una tabla de tamano 40 x 40 x 3. Vamos a analizar con tablas de tamano 20 x 20 x 3 y 80 x 80 x 3

    Tabla 20 x 20 (learning rate = 0.1 y discount rate 0.95)
 ![QLearning_4](qtable_charts/Figure_25000-01-095-tabla20x20.png)    


    Tabla 80 x 80 (learning rate = 0.1 y discount rate 0.95)
 ![QLearning_4](qtable_charts/Figure_25000-01-095-tabla80x80.png)    


 ![QLearning_1_1](qtable_charts/24000.png)

 Con una tabla de 80 x 80 celdas, vemos que en las filas cercanas a la 80 (posición cercana al objetivo), la acción mas elegida es la 2, acelerar a la derecha. En posiciones cercanas a la izquierda (entre las filas 0 y 30 aprox) la acción mas tomada es la 0 o impulso a la izquierda. Tiene sentido porque de esta manera el carro coge impulso para luego subir.


 ### Epsilon

 Hasta ahora, el hiperparametro épsilon tenia una tasa de decaimiento que venia con la formula

    epsilon / (episodios – 1)
Épsilon lo hemos definido =1, por lo tanto la tasa de decaimiento era de 0.00004

Epsilon nos sirve para controlar la exploración frente a la explotación de resultados. Cuando esta cercano a 1, el algoritmo explora por el espacio de estados y genera muchos valores aleatorios. Vamos a hacer que épsilon decaiga mas rápidamente.


    Tabla 40 x 40 (learning rate = 0.1 y discount rate 0.95), epsilon decay 0.00008 (doblamos frente al mejor resultado)



![QLearning_4](qtable_charts/Figure_25000-01-095-tabla40x40-decay12500.png)      

Podemos observar la importancia de esta variable, que hace que nuestro algoritmo comience a encontrar resultados a partir del episodio 6000 aprox. frente a los mas de 10000 que necesitaba anteriormente
Asi mismo los resultados son mejores que antes, mejorando los rewards cerca de los 1500 episodios. 
Con épsilon estamos haciendo que el algoritmo comience antes a explotar los resultados que tenemos en la tabla Q


    Tabla 40 x 40 (learning rate = 0.1 y discount rate 0.95), epsilon decay 0.00016



![QLearning_4](qtable_charts/Figure_25000-01-095-tabla40x40-decay00016.png)     


    Tabla 40 x 40 (learning rate = 0.1 y discount rate 0.95), epsilon decay 0.004



![QLearning_4](qtable_charts/Figure_25000-01-095-tabla40x40-decay004.png)    



### Conclusiones

En este algoritmo y para este entorno, hemos visto que la variable epsilon condiciona mucho los resultados, haciendo que el algoritmo converja rapidamente cuando la tasa de decaimiento es mayor.

