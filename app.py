import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from PIL import Image
stars= pd.read_csv("stars.zip")
stars.dropna(inplace=True)
asu = stars.copy()


###Machine Learning

# Train test split
from sklearn.model_selection import train_test_split

X1 = stars.drop('luminosity_class', axis=1)
y1 = stars['luminosity_class']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.75, random_state=30)
# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X1)
scaler.transform(X1_train)
scaler.transform(X1_test)

# DecisionTreeClassifier model
from sklearn import tree

modelDTC = tree.DecisionTreeClassifier()
modelDTC.fit(X1_train, y1_train)
modelDTC.fit(X1_train, y1_train)

train_pred1 = modelDTC.predict(X1_train)
test_pred1 = modelDTC.predict(X1_test)

# Score
from sklearn.metrics import accuracy_score

print(f'Train score: {accuracy_score(y1_train, train_pred1)}')
print(f'Test score: {accuracy_score(y1_test, test_pred1)}')

#####Graphiques

# creer une colonne spectral
spectral_class = []
for i in asu['bv_color']:
  if i < -0.3:
    spectral_class.append('O')
  elif i >= 1.4:
    spectral_class.append('M')
  elif i < -0.02 and i >= -0.3:
    spectral_class.append('B')
  elif i < 0.3 and i >= -0.02:
    spectral_class.append('A')
  elif i < 0.58 and i >= 0.3:
    spectral_class.append('F')
  elif i < 0.81 and i >= 0.58:
    spectral_class.append('G')
  else:
    spectral_class.append('K')
asu['spectral_class'] = spectral_class


### Texte

st.set_page_config(
    layout = 'wide'
)

st.title('Datathon août 2021 - équipe Datastellar')
st.write(" ")
image = Image.open('galaxy.jfif')
st.image(image)
st.write("")
st.write("Dans le cadre du Datathon organisé par la ** Wild Code School ** en août 2021, nous avons décidé de monter une équipe pour répondre au challenge suivant: "
         "Vulgariser les problématiques complexes des données concernant l’espace auprès du grand public.")
st.write("🚀 Notre équipe est composée de Amazigh, Minh, Philippe, Samir et Steeven. ")
st.write("Nous avons tenté de répondre à une problématique précise qui est de ** prédire le type d'étoile ** ⭐ en fonction de variables numériques. ")
st.write("Pour cela nous avons dans un premier collecter et nettoyer un jeu de donnée, avant de passer à l'analyse de ces mêmes données. Enfin nous avons tenté de répondre à la problématique en créant un modèle de Machine Learning.")
st.write("Pour présenter le résultat de notre travail nous avons donc décider de créer 3 sections différentes dans l'application, correspondant chacune aux étapes citées précédemment.")
st.write(" ")
st.write(" ")
st.write(" ")

## Liste déroulante

sel_col, disp_col = st.columns(2)
selectbox1 = sel_col.selectbox('Veuillez choisir la section qui vous intéresse:', options = ['Présentation du dataset','Analyse', 'Machine Learning'])

## Graphiques

if selectbox1 == 'Présentation du dataset':
    st.write('')
    st.write('')
    st.write(" Dans notre dataset, les types d'étoiles sont représentés dans la variable lumonosity_class via un indice (voir ci-dessous).")
    st.write("Il existe 6 types d'étoiles: ")
    st.markdown('* ** 1 : ** Luminous supergiants (0.6% du dataset)')
    st.markdown('* ** 15  : ** Underluminous supergiants (0.2% du dataset)')
    st.markdown('* ** 13  : ** Bright giants (1.3% du dataset)')
    st.markdown('* ** 3  : ** Giants (8.9% du dataset)')
    st.markdown('* ** 35  : ** Subgiants (3.1% du dataset)')
    st.markdown('* ** 5 : ** Dwarfs (main-sequence stars, 85.8% du dataset)')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    graph1, graph2 = st.columns(2)

    with graph2:
        st.header('Aperçu du dataset')
        st.write(stars.head(8))

    with graph1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('Le dataset que nous avons choisi pour procéder à notre analyse regroupe des données sur la typologie des étoiles.')
        st.write(stars.shape, ':  Après une étape indispensable de nettoyage, nous avons décidé de garder 7 colonnes et près de 9500 lignes. ')
        st.write("Vous trouverez le dataset original en cliquant sur le lien suivant: ")
        st.write('http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=V/19/data')


    graph5, graph4 = st.columns(2)

    with graph4:
        st.header('Exemple de caractéristique: Répartition de la masse des étoiles')
        st.line_chart(stars['mass'])

    with graph5:
        st.write('')
        st.header('Description statistique des variables quantitatives')
        st.write('')
        st.write('')
        st.write(stars.describe())


if selectbox1 == 'Analyse':
    st.write('')
    st.write('')
    graph5, graph4 = st.columns(2)

    with graph5:
        stars['luminosity_class'] = stars['luminosity_class'].astype('category')
        f, axes = plt.subplots(2, 3, figsize=(18, 10))
        sns.despine(left=True)

        sns.distplot(stars['temperature'], color='b', ax=axes[0, 0])
        sns.distplot(stars['luminosity'], color='m', ax=axes[0, 1])
        sns.distplot(stars['magnitude'], color='r', ax=axes[0, 2])
        sns.distplot(stars['bv_color'], color='g', ax=axes[1, 0])
        sns.distplot(stars['age'], color='r', ax=axes[1, 1])
        sns.distplot(stars['mass'], color='b', ax=axes[1, 2])

        plt.setp(axes, yticks=[])
        plt.tight_layout()
        st.write(f)

    with graph4:
        data1 = stars.groupby(by=['luminosity_class']).count()
        data1.head()
        data = stars.groupby("luminosity_class")['luminosity_class'].value_counts()
        lab = {'Dwarfs (main-sequence stars)': 8137, 848: 'Giants(13)', 293: 'Subgiants', 120: 'Bright giants',
               60: 'Luminous supergiants', 21: 'Underluminous supergiants'}
        data = stars.groupby(by=['luminosity_class']).count()
        data['Luminosity Class'] = data.index
        data['Luminosity Class'] = data['Luminosity Class'].map(
            {5: 'Dwarfs (main-sequence stars)', 3: 'Giants', 35: 'Subgiants', 13: 'Bright giants',
             1: 'Luminous supergiants', 15: 'Underluminous supergiants'})
        data.head()

        fig = px.pie(data, values='luminosity', names='Luminosity Class', title='Distribution of Luminosity classes',
                     labels=lab)
        st.write(fig)
    graph1, graph2 = st.columns(2)
    with graph2:
        fig = px.scatter(asu,
                             x='temperature',
                             y='magnitude',
                             color='spectral_class',
                             color_discrete_map={'O': '#4684c2',
                                                 'B': '#64d9e8',
                                                 'A': '#a5ecf0',
                                                 'F': '#edfeff',
                                                 'G': '#e8f29d',
                                                 'K': '#f2d95a',
                                                 'M': '#f7300c'},
                             category_orders={'spectral_class': ['O', 'B', 'A', 'F', 'G', 'K', 'M']},
                             title='The Hertzsprung-Russell diagram')
        fig['layout']['xaxis']['autorange'] = "reversed"
        # transparent background
        # fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.write(fig)

    with graph1:
        sun = asu.groupby(['spectral_class', 'luminosity_class'])['bv_color'].count().reset_index()
        sun = sun[sun['bv_color'] != 0]
        sun['luminosity_class'] = sun['luminosity_class'].map({
            5: 'Main-sequence',
            3: 'Giants',
            35: 'Subgiants',
            13: 'Bright giants',
            1: 'Luminous supergiants',
            15: 'Underluminous supergiants'})

        fig2 = px.sunburst(sun,
                           path=['luminosity_class', 'spectral_class'],
                           values='bv_color',
                           title='Frequency of Luminosity Classes',
                           hover_name='luminosity_class')
        st.write(fig2)

    st.write(' ')
    st.write(' ')
    graph1, graph2 = st.columns(2)
    with graph1:
        data2 = data = stars.groupby(by=['luminosity_class']).median()
        data2['Luminosity Class'] = data.index
        data['Luminosity Class'] = data['Luminosity Class'].map(
            {5: 'Dwarfs (main-sequence stars)', 3: 'Giants', 35: 'Subgiants', 13: 'Bright giants',
             1: 'Luminous supergiants', 15: 'Underluminous supergiants'})

        fig, axes = plt.subplots(6, 1, figsize=(15, 30))
        fig.suptitle('Comparing Luminosity Classes on Median for each Parameter', size=18)
        fig.tight_layout(pad=4.0)

        sns.barplot(ax=axes[1], x="Luminosity Class", y="temperature", palette='inferno', data=data2)
        sns.barplot(ax=axes[0], x="Luminosity Class", y="luminosity", palette='inferno', data=data2)
        sns.barplot(ax=axes[2], x="Luminosity Class", y="magnitude", palette='inferno', data=data2)
        sns.barplot(ax=axes[3], x="Luminosity Class", y="bv_color", palette='inferno', data=data2)
        sns.barplot(ax=axes[4], x="Luminosity Class", y="age", palette='inferno', data=data2)
        sns.barplot(ax=axes[5], x="Luminosity Class", y="mass", palette='inferno', data=data2)

        st.write(fig)
    with graph2 :
        fig = plt.figure(figsize=(20, 8))
        sns.scatterplot(data=stars, x='bv_color', y='temperature',
                        hue="luminosity_class", palette='Paired').set_title('Temperature by BV Color')
        st.write(fig)

## Partie ML

if selectbox1 == 'Machine Learning':
    st.write("Pour définir notre modèle de Machine Learning, nous avons fait plusieurs essais et sommes arrivés à la conclusion que le type d'algorithme le plus adapté était un ** arbre de décision **." )
    st.write(" Si on se réfère au score de précision, notre modèle permet de trouver la bonne étoile dans", round(accuracy_score(y1_test, test_pred1) *100,2), '% des cas.')
    st.write("Voici les 6 variables X, qui permettent de définir à quelle classe appartiennent les étoiles:")
    st.write(X1)
    st.write('')
    st.write('')
    st.write(" En fonction des données qu'on lui indique pour chaque colonne (voir tableau ci-dessus), le modèle prédit à quel type suivant appartient l'étoile: ")
    st.markdown('* ** Ia : ** Luminous supergiants')
    st.markdown('* ** Ib : ** Underluminous supergiants')
    st.markdown('* ** II : ** Bright giants')
    st.markdown('* ** III : ** Giants')
    st.markdown('* ** IV : ** Subgiants')
    st.markdown('* ** V : ** Dwarfs (main-sequence stars)')