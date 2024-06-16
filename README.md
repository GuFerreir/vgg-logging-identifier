## BACKEND/SERVIDOR DA APLICAÇÃO
API que identifica áreas desmatadas

Tópicos Avançados em Banco de Dados (1 Semestre 2024)

## COMO USAR

### 1 - Gere o modelo (como o modelo é muito pesado, não da pra subir no github)
- Primeiro garanta que tem todas as dependencias instaladas (use o ChatGPT pra ver quais são e como instalar)

- Segundo use o comando **python .\train_model.py**


### 2 - Com o modelo gerado, agora é hora de subir o servidor
- Primeiro garanta que tem todas as dependencias instaladas (use o ChatGPT pra ver quais são e como instalar)

- Segundo use o comando **python .\app.py**

- Para testar se deu tudo certo, abra o navegador e acesse **http://localhost:3000/** (Você deverá recer a mensagem "Backend rodando")


### 3 - Com o servidor rodando siga os passos abaixo
- **Passo 3.1:** Abra o Postman.

- **Passo 3.2:** Crie uma nova Request e Selecione o método POST.

- **Passo 3.3:** Digite http://127.0.0.1:3000/api/prediction na URL.

- **Passo 3.4:** Vá para a aba "Body".

- **Passo 3.5:** Selecione "form-data".

- **Passo 3.6:** Adicione um campo com o nome image e selecione o arquivo de imagem que você deseja enviar. (imagens de test podem ser encontradas no diretório do projeto em /images/test/)

- **Passo 3.7:** Clique em "Send".