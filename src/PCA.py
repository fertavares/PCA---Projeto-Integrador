import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
                
    def LimparData(self):
        if self.data is not None:
            try:
                # self.acessibilidade = self.data.loc[:, "QT_DOC_ESP":"QT_DOC_ESP_CE"].dropna().drop_duplicates().reset_index(drop=True)
                self.acessibilidade = self.data[["QT_SALAS_UTILIZADAS","QT_EQUIP_SOM","QT_EQUIP_TV"]].dropna().drop_duplicates().reset_index(drop=True)

            except KeyError:
                print("[Erro] As Colunas Especificadas não foram encontradas")          
        
    def CalcularPca(self):
        if hasattr(self, 'acessibilidade'):
            
            scaler = StandardScaler() # Padronizando os dados
            scaled_data = scaler.fit_transform(self.acessibilidade)
            
            pca = PCA(n_components=2) # Calcula o PCA
            pca_results = pca.fit_transform(scaled_data)
            
            self.pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2']) # Cria um DataFrame com os resultados do PCA
            
            colunas_acessibilidade = self.acessibilidade.columns
            self.pca_df['Categoria Acessibilidade'] = self.acessibilidade[colunas_acessibilidade].apply(
                lambda row: ', '.join([col for col in colunas_acessibilidade if row[col] == 1]), axis=1
            ) # Rever
        else:
            print("[Erro] Não há dados para realizar a PCA")
        
    def PlotPca(self):
        if hasattr(self, 'pca_df'):
            plt.figure(figsize=(12, 8))
            
            sns.scatterplot(x='PC1', y='PC2', hue='Categoria Acessibilidade', data=self.pca_df, palette='tab10')
            
            plt.title('PCA das Colunas de Acessibilidade das Escolas')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            
            plt.legend(title='Legenda de Acessibilidade', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)# Ajusta a legenda para mostrar múltiplas colunas
            plt.tight_layout()  # Evitar sobreposição
            plt.show()
        else:
            print("[Erro] O PCA não foi calculado. Verifique os dados.")

if __name__ == "__main__":
    caminho = r"..\TrabalhoPCA\DataFrame\DFmicrodados2021.csv"
    pca = PCAAcessibilidade(caminho)
    # colunas = [
    #     "IN_ACESSIBILIDADE_CORRIMAO", "IN_ACESSIBILIDADE_ELEVADOR", 
    #     "IN_ACESSIBILIDADE_PISOS_TATEIS", "IN_ACESSIBILIDADE_VAO_LIVRE", 
    #     "IN_ACESSIBILIDADE_RAMPAS", "IN_ACESSIBILIDADE_SINAL_SONORO", 
    #     "IN_ACESSIBILIDADE_SINAL_TATIL", "IN_ACESSIBILIDADE_SINAL_VISUAL", 
    #     "IN_ACESSIBILIDADE_INEXISTENTE"
    # ]
    pca.LimparData()  # Passando as colunas específicas
    pca.CalcularPca()
    pca.PlotPca()
