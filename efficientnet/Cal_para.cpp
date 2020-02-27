#include <iostream>
#include <cmath>
using namespace std;
string itor(int num, int mi, string dic);
string intToRoman(string num);
int whichBrackets(char ch);

int main()
{
    // 输入 baseline 参数和 α/β 参数
    int b0_Ci[7] = {1,2,2,3,3,4,1};
    float depth[8] = {1.0,1.1,1.2,1.4,1.8,2.2,2.6,3.1};
    float width[8] = {1.0,1.0,1.1,1.2,1.4,1.6,1.8,2.0};
    int width0 = 320;
    // 读入 bi
    int index;
    cout << "Enter the aim bi: ";
    cin >> index;
    // 计算并输出深度和宽度
    int bi_ci[7] = {1,2,2,3,3,4,1};
    for (int i = 0; i < 7; i++) bi_ci[i] = ceil(bi_ci[i] * depth[index]);
    int layer1 = 0;
    int layer2 = 0;
    int layer3 = 0;
    int widthi = width0 * width[index];
    for (int i = 0; i < 3; i++) layer1 += bi_ci[i];
    for (int i = 0; i < 5; i++) layer2 += bi_ci[i];
    for (int i = 0; i < 7; i++) layer3 += bi_ci[i];
    cout << "INDICES are : " << layer1-1 << " " << layer2-1 << " " << layer3-1 << endl;
    cout << "EXTRAS[0] are : (" << widthi << ", 128, 1, 1, 0)" << endl;
    return 0;
}
