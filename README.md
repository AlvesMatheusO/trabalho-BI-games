# 🎮 VGChartz Dashboard Estratégico

Dashboard analítico desenvolvido com **Python, Flask, Pandas e ApexCharts**, utilizando dados reais da base **VGChartz 2024** para análise de vendas, gêneros, plataformas, publishers e oportunidades estratégicas na indústria de jogos.

---

# 📋 Funcionalidades

O dashboard fornece:

* 📊 KPIs estratégicos

  * Total de jogos
  * Vendas totais
  * Nota média da crítica
  * Quantidade de consoles

* 🎮 Top Consoles por vendas

* 🎯 Top Gêneros por vendas

* 🌎 Distribuição regional das vendas

* 🕹️ Market Share dos consoles

* 🏢 Ranking de publishers

* ⭐ Distribuição de notas da crítica

* 📈 Correlação entre crítica e vendas

* 🧭 Radar estratégico dos gêneros

* 💡 Insights automáticos gerados pelo backend

---

# 🛠️ Tecnologias Utilizadas

### Backend

* Python 3.10+
* Flask
* Flask-CORS
* Pandas
* NumPy

### Frontend

* HTML5
* CSS3
* JavaScript
* ApexCharts

---

# 📁 Estrutura do Projeto

```txt
dashboard_jogos/

├── app.py
├── index.html
├── vgchartz-2024.csv
└── README.md
```

---

# 🚀 Instalação

## 1. Clonar ou baixar o projeto

Extraia os arquivos para uma pasta local.

Exemplo:

```txt
dashboard_jogos/
```

---

## 2. Abrir terminal na pasta do projeto

Windows:

```powershell
cd C:\Users\SeuUsuario\Downloads\dashboard_jogos
```

Linux:

```bash
cd ~/dashboard_jogos
```

---

## 3. Criar ambiente virtual (Opcional)

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Instalar dependências

```bash
pip install flask flask-cors pandas numpy
```

---

# ▶️ Executando o Projeto

Inicie o servidor Flask:

```bash
python app.py
```

Se tudo estiver correto, será exibido:

```txt
* Running on http://127.0.0.1:5000
```

---

# 🌐 Acessando o Dashboard

Abra o navegador e acesse:

```txt
http://127.0.0.1:5000
```

---

# ✅ Verificando a API

Você pode testar o funcionamento da API acessando:

### Status da aplicação

```txt
http://127.0.0.1:5000/api/health
```

Retorno esperado:

```json
{
  "status": "ok",
  "rows": 64016,
  "csv": "vgchartz-2024.csv"
}
```

---

### Lista de gêneros

```txt
http://127.0.0.1:5000/api/genres
```

---

### Resumo geral

```txt
http://127.0.0.1:5000/api/summary
```

---

# ⚠️ Problemas Comuns

## Dashboard em branco

Verifique se o Flask está rodando:

```bash
python app.py
```

---

## Erro ao carregar CSV

Confirme que o arquivo:

```txt
vgchartz-2024.csv
```

está na mesma pasta do:

```txt
app.py
```

---

## Erro "ModuleNotFoundError"

Instale as dependências:

```bash
pip install flask flask-cors pandas numpy
```

---

## Erro ao abrir o HTML

Não abra o arquivo HTML com duplo clique.

❌ Incorreto:

```txt
file:///C:/...
```

✅ Correto:

```txt
http://127.0.0.1:5000
```

O HTML deve ser servido pelo Flask.

---

# 📊 Fonte dos Dados

Dataset:

VGChartz 2024

Contém informações sobre:

* Jogos
* Consoles
* Publishers
* Desenvolvedoras
* Vendas globais
* Vendas regionais
* Notas da crítica

---

# 👨‍💻 Autor
Guilherme Celestino - 2210465
João Mateus - 2220305
Maria Iana - 2219236
Matheus Alves - 2217227


Projeto desenvolvido para análise estratégica do mercado de videogames utilizando Business Intelligence, Visualização de Dados e Analytics.
