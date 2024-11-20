import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class PCAAcessibilidade:
    
    def __init__(self, caminho):
        try:
            self.data = pd.read_csv(caminho, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(caminho, encoding="ISO-8859-1", low_memory=False)
            except Exception as e:
                self.data = None
                print("[Erro] Não foi possível carregar o arquivo:", e)
                
    def limpar_data(self):
        if self.data is not None:
            try:
            
                self.acessibilidade = self.data[["NO_BAIRRO", "QT_SALAS_UTILIZADAS_ACESSIVEIS", "QT_TUR_ESP"]].dropna().drop_duplicates().reset_index(drop=True)

                self.acessibilidade['NO_BAIRRO_NUMERICO'], categorias = pd.factorize(self.acessibilidade['NO_BAIRRO'])
                print(self.acessibilidade.head())  
            except KeyError:
                print("[Erro] As colunas especificadas não foram encontradas.")          
        
    def calcular_pca(self):
        if hasattr(self, 'acessibilidade'):
            
            colunas_pca = self.acessibilidade[["NO_BAIRRO_NUMERICO", "QT_SALAS_UTILIZADAS_ACESSIVEIS", "QT_TUR_ESP"]]

            self.scaler = StandardScaler()
            colunas_pca_scaled = self.scaler.fit_transform(colunas_pca)

            self.cov_matriz = np.cov(colunas_pca_scaled, rowvar=False)
            print("Matriz de Covariância:\n", self.cov_matriz)

            self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matriz)
            print("Autovalores:\n", self.eigenvalues)
            print("Autovetores:\n", self.eigenvectors)

            self.pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
            pca_results = self.pca_pipeline.fit_transform(colunas_pca)

            self.pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
            self.pca_df['NO_BAIRRO_NUMERICO'] = self.acessibilidade['NO_BAIRRO_NUMERICO']
            self.pca_df['NO_BAIRRO'] = self.acessibilidade['NO_BAIRRO']

            print(self.pca_df.info())

        else:
            print("[Erro] Não há dados para realizar a PCA.")
    
    def plot_pca(self):
        if hasattr(self, 'pca_df'):
            plt.figure(figsize=(18, 12)) 
            
            sns.scatterplot(
                x='PC1', y='PC2',
                hue='NO_BAIRRO',
                palette='tab20', 
                data=self.pca_df,
                marker="o",
                edgecolor="k",
                s=100
            )

            
            pca = self.pca_pipeline.named_steps['pca']
            coeff = np.transpose(pca.components_)

            for i in range(len(coeff)):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='red', alpha=0.5, head_width=0.05, head_length=0.1)
                plt.text(coeff[i, 0] * 1.5, coeff[i, 1] * 1.5, f"Componente {i+1}", color='red', ha='center', va='center')

            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.title('PCA das Colunas de Acessibilidade das Escolas por Bairro')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')

            plt.legend(title='Bairros', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.show()
        else:
            print("[Erro] O PCA não foi calculado. Verifique os dados.")

if __name__ == "__main__":
    caminho = r"../TrabalhoFaculdade-main/DataFrame/DFmicrodados2021.csv"
    pca = PCAAcessibilidade(caminho)
    pca.limpar_data()
    pca.calcular_pca()
    pca.plot_pca()
