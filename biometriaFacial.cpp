#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

int im_width;
int im_height;

// funcao usada para ler as imagens usadas no treino do classificador
static void read_csv(const string& nomeArquivo, vector<Mat>& imagens, vector<int>& classes, char separador = ';') {
    std::ifstream arquivo(nomeArquivo.c_str(), ifstream::in);
    string linha, caminho, classe;

    if (!arquivo) {
        CV_Error(CV_StsBadArg, "Arquivo invalido, por favor verifique o nome do arquivo.");
    }

    while (getline(arquivo, linha)) {
        stringstream linhas(linha);
        getline(linhas, caminho, separador);
        getline(linhas, classe);

        if(!caminho.empty() && !classe.empty()) {
            Mat foto = imread(caminho, CV_LOAD_IMAGE_COLOR); // primeiro le a imagem colorida para um buffer porque ela pode nao ser uma matriz continua
            Mat fotoCinza;
            cvtColor(foto, fotoCinza, CV_BGR2GRAY); // converte para cinza
            imagens.push_back(fotoCinza);
            classes.push_back(atoi(classe.c_str()));
        }
    }
}

void encontrarEMostrarFaces(Mat& frame, const string nomeJanela, CascadeClassifier& detector, Ptr<LBPHFaceRecognizer>& classificador, char *nomes[], int* contagem, Mat& destravado, Mat&destravadoAlpha, Mat& travado, Mat& travadoAlpha){
    int i, classificacao, pos_x, pos_y, novo_int;
    double confianca;
    Mat cinza, face, face_reduzida;
    Rect faceAtual;
    string texto;
    vector<Rect_<int> > faces;
if(frame.cols == 0)cout << "frame ta vazio" << endl;
    // converte o frame para cinza
    cvtColor(frame, cinza, CV_BGR2GRAY);

    // encontra as faces no frame e coloca suas posicoes em um vetor de retangulos
    detector.detectMultiScale(cinza, faces, 1.1, 5);

    // iteramos sobre as faces encontradas para classifica-las
    for(i = 0; i < faces.size(); i++) {
        // as faces sao processadas uma a uma
        faceAtual = faces[i];

        // recorta a face da imagem para poder transforma-la livremente
        face = cinza(faceAtual);

        // colocamos a face no mesmo tamanho que as imagens de treino
        //cv::resize(face, face_reduzida, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

        // classificamos a face
        classificacao = -1;
        confianca = 0.0;
        classificador->predict(face, classificacao, confianca);

        // se a confianca for menor do que 5000, entao desenhamos suas informacoes no frame
        if(confianca < 8000){
            // mudamos o tamanho do retangulo da face para corresponder a posicao da face de treino
            novo_int = (int)(faceAtual.width * 0.14);
            faceAtual.x += novo_int;
            faceAtual.width -= 2 * novo_int;
            novo_int = (int)(faceAtual.height * 0.17);
            faceAtual.y += novo_int;
            faceAtual.height -= novo_int;
            // desenhamos um retangulo verde ao redor da face
            rectangle(frame, faceAtual, CV_RGB(0, 255, 0), 1);

            // criamos o texto informativo que ficara acima do retangulo verde
            texto = format("classificacao = %s, %lf", nomes[classificacao], confianca);

            // calculamos a posicao onde o texto iniciara, nao podem ser valores negativos
            pos_x = std::max(faceAtual.tl().x - 10, 0);
            pos_y = std::max(faceAtual.tl().y - 10, 0);

            // escrevemos o texto no frame
            putText(frame, texto, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);

            // verificamos se a classificacao foi correta (estamos tentando encontrar a face da classe 0)
            if(classificacao == 0){
                (*contagem)++;
            }
            else{
                (*contagem)--;
            }

            // verificamos se a contagem total de classificacoes corretas e maior que 10
            if(*contagem > 10){
                destravado.copyTo(frame(Rect(0, 0, destravado.cols, destravado.rows)), destravadoAlpha); // se for, entao destrava o cadeado
            }
            else{
                travado.copyTo(frame(Rect(0, 0, travado.cols, travado.rows)), travadoAlpha); // senao, o mantem travado
            }
        }
        else{
            *contagem = 0;
        }
    }

    // mostra o frame ao usuario
    imshow(nomeJanela, frame);
}

int main(int argc, const char *argv[]) {
    char *nomes[] = {"Bruno\0", "Paula\0"};
    int contagemAcerto = 0;
    int dispositivo; // o id da camera
    Mat travado, destravado; // imagens de cadeados travados e destravados usados para representar se a autenticacao foi um sucesso ou nao
    Mat travadoAlpha, destravadoAlpha; // mascaras para as imagens de cadeado
    Mat original; // a matriz que armazenara o frame lido da camera
    CascadeClassifier detector; // o haar cascade usado para detectar as posicoes dos rostos nas imagens
    string classificadorHaar, arquivoCsv;
    vector<Mat> imagens; vector<int> classes; // vetores para armazenar as imagens e suas classes
    vector<cv::Mat> canais;

    travado = imread("./travado.png", CV_LOAD_IMAGE_COLOR);
    destravado = imread("./destravado.png", CV_LOAD_IMAGE_COLOR);

    resize(travado, travado, Size(138, 188), 1.0, 1.0, INTER_CUBIC);
    resize(destravado, destravado, Size(138, 188), 1.0, 1.0, INTER_CUBIC);

    split(travado, canais);
    travadoAlpha = canais[2].clone();
    //cvtColor(travado, travado, CV_BGRA2BGR);

    split(destravado, canais);
    destravadoAlpha = canais[1].clone();
    //cvtColor(destravado, destravado, CV_BGRA2BGR);

    // verifica se os parametros passados ao programa estao na quantidade
    // esperada, se nao estiver entao imprime como e quais parametros passar
    if (argc != 4) {
        cout << "uso: " << argv[0] << " </caminho/para/haar_cascade> </caminho/para/csv.ext> </caminho/para/device id>" << endl;
        cout << "\t </caminho/para/haar_cascade> -- Caminho para o Haar Cascade para detectar faces." << endl;
        cout << "\t </caminho/para/csv.ext> -- Caminho para o arquivo CSV com as faces de treino." << endl;
        cout << "\t <dispositivo id> -- O id de uma camera de onde lera frames para analisar." << endl;
        exit(EXIT_FAILURE);
    }

    // pega o caminho para o classificador Haar
    classificadorHaar = string(argv[1]);

    // pega o caminho para o CSV
    arquivoCsv = string(argv[2]);

    // pega o id da camera
    dispositivo = atoi(argv[3]);

    // tenta ler as fotos de treino
    try {
        read_csv(arquivoCsv, imagens, classes);
    } catch (cv::Exception& e) { // se falhar, entao avisa o usuario e termina a execucao
        cerr << "Erro ao tentar ler arquivos de treino do \"" << arquivoCsv << "\". Mensagem de erro: " << e.msg << endl;
        exit(1);
    }

    // pega as dimensoes das imagens de treino
    im_width = imagens[0].cols;
    im_height = imagens[0].rows;

    // sera usado um Fisher face recognizer
    //Ptr<FisherFaceRecognizer> classificador = FisherFaceRecognizer::create();
    // tambem podemos usar o Eigen face recognizer, mas nao mostrou-se tao acertivo quanto o Fisher
    //Ptr<EigenFaceRecognizer> classificador = EigenFaceRecognizer::create();
    // tambem podemos usar o Local ninary patterns histogram
    Ptr<LBPHFaceRecognizer> classificador = LBPHFaceRecognizer::create();

    // treinamos o classificador com as imagens de treino
    classificador->train(imagens, classes);

    // carrega o haar_cascade passado como argumento
    detector.load(classificadorHaar);

    // ativa a camera pois esta pronto para coletar frames para analisar
    VideoCapture cap(dispositivo);

    // verifica se a camera foi ativada com sucesso
    if(cap.isOpened()) {
        // le frames da camera ate o usuario pressionar q ou ESC
        while(1) {
            // le um frame da camera
            cap >> original;
/*
            // testes com equalizacao de luminosidade do frame, nao deram bom resultado
            Mat YCrCbframe;

            cvtColor(frame, YCrCbframe, CV_RGB2YCrCb);
            split(YCrCbframe, canais);
            equalizeHist(canais[0], canais[0]);
            merge(canais, YCrCbframe);
            cvtColor(YCrCbframe, YCrCbframe, CV_YCrCb2BGR);

            cvtColor(frame, frame, CV_BGR2YUV);
            split(frame, canais);
            equalizeHist(canais[0], canais[0]);
            merge(canais, frame);
            cvtColor(frame, frame, CV_YUV2BGR);

            //encontrarEMostrarFaces(frame, "luz normalizada YUV", detector, classificador, nomes, contagemAcerto, destravado, destravadoAlpha, travado, travadoAlpha);

            //encontrarEMostrarFaces(YCrCbframe, "luz normalizada YCrCb", detector, classificador, nomes, contagemAcerto, destravado, destravadoAlpha, travado, travadoAlpha);
*/
            // chama a rotina encarregada por encontrar e classificar as faces no frame
            encontrarEMostrarFaces(original, "original", detector, classificador, nomes, &contagemAcerto, destravado, destravadoAlpha, travado, travadoAlpha);

            // espera o usuario pressionar uma tecla por 5ms
            char tecla = (char) waitKey(5);

            // se q ou ESC foi pressionado, entao termina a execucao do programa
            if(tecla == 27 || tecla == 'q')
                break;
        }
    }
    else{ // se nao foi, entao avisa o usuario e termina a execucao
        cerr << "Capture Device ID " << dispositivo << "cannot be opened." << endl;
        return -1;
    }

    // desativa a camera
    cap.release();

    // fecha as janelas abertas
    destroyAllWindows();
    return 0;
}