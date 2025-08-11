from PIL import Image
import numpy as np

def converter_para_cinza(imagem_rgb):
    """
    Converte uma imagem RGB (3 canais) para níveis de cinza (1 canal).
    Utiliza a fórmula de luminância ponderada: Gray = 0.299*R + 0.587*G + 0.114*B.
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_cinza = np.zeros((altura, largura), dtype=np.uint8)

    # Itera sobre cada pixel e aplica a fórmula de luminância
    for i in range(altura):
        for j in range(largura):
            r, g, b = imagem_rgb[i, j]
            # Aplica a fórmula de conversão
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            imagem_cinza[i, j] = gray

    return imagem_cinza

def converter_para_binario(imagem_cinza, limiar=127):
    """
    Converte uma imagem em níveis de cinza para binário, usando um limiar.
    Pixels > limiar se tornam 255 (branco), caso contrário se tornam 0 (preto).
    """
    # A imagem_cinza é um array numpy com shape (altura, largura)
    altura, largura = imagem_cinza.shape
    imagem_binaria = np.zeros((altura, largura), dtype=np.uint8)

    # Itera sobre cada pixel e aplica o limiar
    for i in range(altura):
        for j in range(largura):
            if imagem_cinza[i, j] > limiar:
                imagem_binaria[i, j] = 255  # Branco
            else:
                imagem_binaria[i, j] = 0   # Preto

    return imagem_binaria

if __name__ == "__main__":
    nome_arquivo = 'wolf-8142720_1280.png'

    try:
        # Carrega a imagem usando Pillow e a converte para um array numpy
        imagem_original_pil = Image.open(nome_arquivo).convert('RGB')
        imagem_original_np = np.array(imagem_original_pil)

        # Chama nossa função para converter para cinza
        imagem_cinza_np = converter_para_cinza(imagem_original_np)

        # Chama nossa função para converter para binário
        # Podemos ajustar o limiar (threshold) conforme a necessidade
        imagem_binaria_np = converter_para_binario(imagem_cinza_np, limiar=120)

        # Converte os arrays numpy de volta para objetos de imagem PIL
        imagem_cinza_pil = Image.fromarray(imagem_cinza_np)
        imagem_binaria_pil = Image.fromarray(imagem_binaria_np)

        # Salva as imagens resultantes
        imagem_cinza_pil.save('imagem_cinza_custom.jpg')
        imagem_binaria_pil.save('imagem_binaria_custom.jpg')

        print("Processamento concluído!")
        print("Imagem em níveis de cinza salva como 'imagem_cinza_custom.jpg'")
        print("Imagem binária salva como 'imagem_binaria_custom.jpg'")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
