#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    12.04.2015 20:27:58 CEST
# File:    shapes.py


from common import *

import numpy as np

class ShapesTestCase(CommonTestCase):
    def testsphere(self):
        sphere = z2pack.shapes.Sphere([1, 2, 3], 2)

        points = []
        for t in np.linspace(0, 1, 23):
            for k in np.linspace(0, 1, 27):
                points.append(sphere(t, k))
        self.assertContainerAlmostEqual(
            points, [[1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.2846296765465703, 2.0, 1.0203571162381346], [1.2763588554395162, 2.0681163401186953, 1.0203571162381346], [1.2520270621778848, 2.1322740061425138, 1.0203571162381346], [1.2130483721435719, 2.1887443877257793, 1.0203571162381346], [1.1616880851195983, 2.2342456315523576, 1.0203571162381346], [1.1009310743007501, 2.2661333707213389, 1.0203571162381346], [1.03430831631307, 2.2825544057394391, 1.0203571162381346], [0.96569168368692992, 2.2825544057394391, 1.0203571162381346], [0.89906892569925001, 2.2661333707213389, 1.0203571162381346], [0.8383119148804018, 2.2342456315523576, 1.0203571162381346], [0.78695162785642814, 2.1887443877257793, 1.0203571162381346], [0.74797293782211516, 2.1322740061425138, 1.0203571162381346], [0.72364114456048378, 2.0681163401186953, 1.0203571162381346], [0.71537032345342966, 2.0, 1.0203571162381346], [0.72364114456048378, 1.9318836598813047, 1.0203571162381346], [0.74797293782211527, 1.8677259938574859, 1.0203571162381346], [0.78695162785642814, 1.8112556122742205, 1.0203571162381346], [0.83831191488040169, 1.7657543684476427, 1.0203571162381346], [0.8990689256992499, 1.7338666292786609, 1.0203571162381346], [0.96569168368693015, 1.7174455942605609, 1.0203571162381346], [1.03430831631307, 1.7174455942605609, 1.0203571162381346], [1.1009310743007501, 1.7338666292786609, 1.0203571162381346], [1.1616880851195983, 1.7657543684476427, 1.0203571162381346], [1.2130483721435719, 1.8112556122742207, 1.0203571162381346], [1.2520270621778848, 1.8677259938574862, 1.0203571162381346], [1.2763588554395162, 1.9318836598813047, 1.0203571162381346], [1.2846296765465703, 2.0, 1.0203571162381346], [1.5634651136828595, 2.0, 1.0810140527710053], [1.5470918415354125, 2.1348460279838779, 1.0810140527710053], [1.4989235801558587, 2.261855294966701, 1.0810140527710053], [1.4217596938110715, 2.3736464840113296, 1.0810140527710053], [1.3200846670960953, 2.4637226975549282, 1.0810140527710053], [1.1998074829899199, 2.5268490334800573, 1.0810140527710053], [1.0679182142430201, 2.5593568185976432, 1.0810140527710053], [0.93208178575697975, 2.5593568185976432, 1.0810140527710053], [0.8001925170100801, 2.5268490334800573, 1.0810140527710053], [0.67991533290390471, 2.4637226975549282, 1.0810140527710053], [0.5782403061889283, 2.3736464840113296, 1.0810140527710053], [0.5010764198441412, 2.261855294966701, 1.0810140527710053], [0.45290815846458754, 2.1348460279838779, 1.0810140527710053], [0.43653488631714066, 2.0, 1.0810140527710053], [0.45290815846458765, 1.8651539720161219, 1.0810140527710053], [0.50107641984414131, 1.7381447050332988, 1.0810140527710053], [0.57824030618892852, 1.6263535159886704, 1.0810140527710053], [0.67991533290390449, 1.5362773024450718, 1.0810140527710053], [0.80019251701007998, 1.4731509665199427, 1.0810140527710053], [0.93208178575698009, 1.4406431814023568, 1.0810140527710053], [1.0679182142430201, 1.4406431814023568, 1.0810140527710053], [1.1998074829899199, 1.4731509665199427, 1.0810140527710053], [1.3200846670960957, 1.536277302445072, 1.0810140527710053], [1.4217596938110717, 1.6263535159886706, 1.0810140527710053], [1.4989235801558589, 1.738144705033299, 1.0810140527710053], [1.5470918415354125, 1.8651539720161223, 1.0810140527710053], [1.5634651136828595, 1.9999999999999998, 1.0810140527710053], [1.8308300260037726, 2.0, 1.1807360092909631], [1.8066876154202371, 2.1988306395831416, 1.1807360092909631], [1.7356634528186536, 2.3861059651136802, 1.1807360092909631], [1.6218852043670993, 2.5509422153898909, 1.1807360092909631], [1.4719652482984729, 2.6837597067011121, 1.1807360092909631], [1.2946163875226147, 2.7768395692242729, 1.1807360092909631], [1.1001454931909385, 2.8247723396810622, 1.1807360092909631], [0.89985450680906132, 2.8247723396810622, 1.1807360092909631], [0.70538361247738512, 2.7768395692242729, 1.1807360092909631], [0.52803475170152736, 2.6837597067011121, 1.1807360092909631], [0.37811479563290062, 2.5509422153898909, 1.1807360092909631], [0.26433654718134625, 2.3861059651136798, 1.1807360092909631], [0.19331238457976285, 2.1988306395831416, 1.1807360092909631], [0.16916997399622724, 2.0, 1.1807360092909631], [0.19331238457976296, 1.8011693604168577, 1.1807360092909631], [0.26433654718134647, 1.6138940348863198, 1.1807360092909631], [0.37811479563290074, 1.4490577846101091, 1.1807360092909631], [0.52803475170152714, 1.3162402932988879, 1.1807360092909631], [0.70538361247738501, 1.2231604307757271, 1.1807360092909631], [0.89985450680906176, 1.1752276603189378, 1.1807360092909631], [1.1001454931909387, 1.1752276603189378, 1.1807360092909631], [1.2946163875226147, 1.2231604307757269, 1.1807360092909631], [1.4719652482984731, 1.3162402932988884, 1.1807360092909631], [1.6218852043670995, 1.4490577846101096, 1.1807360092909631], [1.7356634528186539, 1.6138940348863202, 1.1807360092909631], [1.8066876154202371, 1.8011693604168582, 1.1807360092909631], [1.8308300260037726, 1.9999999999999998, 1.1807360092909631], [2.0812816349111953, 2.0, 1.3174929343376376], [2.0498615557500885, 2.2587676327407094, 1.3174929343376376], [1.9574273390602719, 2.5024966312486026, 1.3174929343376376], [1.80935092553105, 2.7170223520492645, 1.3174929343376376], [1.6142379780808782, 2.8898773400190287, 1.3174929343376376], [1.3834277520088525, 3.0110158915594081, 1.3174929343376376], [1.1303340986932437, 3.0733978743755959, 1.3174929343376376], [0.86966590130675603, 3.0733978743755959, 1.3174929343376376], [0.61657224799114729, 3.0110158915594081, 1.3174929343376376], [0.38576202191912201, 2.8898773400190287, 1.3174929343376376], [0.19064907446894985, 2.7170223520492645, 1.3174929343376376], [0.042572660939728002, 2.5024966312486026, 1.3174929343376376], [-0.049861555750088549, 2.2587676327407089, 1.3174929343376376], [-0.081281634911195111, 2.0, 1.3174929343376376], [-0.049861555750088549, 1.7412323672592902, 1.3174929343376376], [0.042572660939728335, 1.497503368751397, 1.3174929343376376], [0.19064907446895007, 1.2829776479507351, 1.3174929343376376], [0.38576202191912179, 1.1101226599809715, 1.3174929343376376], [0.61657224799114707, 0.98898410844059192, 1.3174929343376376], [0.86966590130675669, 0.92660212562440436, 1.3174929343376376], [1.1303340986932437, 0.92660212562440436, 1.3174929343376376], [1.3834277520088525, 0.98898410844059192, 1.3174929343376376], [1.6142379780808787, 1.1101226599809719, 1.3174929343376376], [1.8093509255310503, 1.2829776479507355, 1.3174929343376376], [1.9574273390602719, 1.4975033687513974, 1.3174929343376376], [2.0498615557500885, 1.7412323672592909, 1.3174929343376376], [2.0812816349111953, 1.9999999999999998, 1.3174929343376376], [2.3097214678905704, 2.0, 1.4885008512914835], [2.2716633423555868, 2.3134368631199069, 1.4885008512914835], [2.1597007656710723, 2.6086579150519267, 1.4885008512914835], [1.9803405958265234, 2.868505981342631, 1.4885008512914835], [1.7440065939456144, 3.077879636888496, 1.4885008512914835], [1.4644336331785195, 3.2246108458714673, 1.4885008512914835], [1.157869477798658, 3.3001721237716986, 1.4885008512914835], [0.84213052220134166, 3.3001721237716986, 1.4885008512914835], [0.53556636682148029, 3.2246108458714673, 1.4885008512914835], [0.25599340605438581, 3.077879636888496, 1.4885008512914835], [0.019659404173476447, 2.868505981342631, 1.4885008512914835], [-0.15970076567107272, 2.6086579150519262, 1.4885008512914835], [-0.27166334235558676, 2.3134368631199069, 1.4885008512914835], [-0.3097214678905702, 2.0, 1.4885008512914835], [-0.27166334235558676, 1.6865631368800922, 1.4885008512914835], [-0.15970076567107228, 1.3913420849480731, 1.4885008512914835], [0.019659404173476669, 1.1314940186573685, 1.4885008512914835], [0.25599340605438548, 0.92212036311150425, 1.4885008512914835], [0.53556636682148007, 0.77538915412853293, 1.4885008512914835], [0.84213052220134244, 0.69982787622830145, 1.4885008512914835], [1.1578694777986582, 0.69982787622830145, 1.4885008512914835], [1.4644336331785195, 0.77538915412853293, 1.4885008512914835], [1.7440065939456151, 0.92212036311150469, 1.4885008512914835], [1.9803405958265239, 1.1314940186573692, 1.4885008512914835], [2.1597007656710727, 1.3913420849480738, 1.4885008512914835], [2.2716633423555868, 1.6865631368800929, 1.4885008512914835], [2.3097214678905704, 1.9999999999999998, 1.4885008512914835], [2.5114991487085163, 2.0, 1.6902785321094298], [2.4675777304849777, 2.3617254228432567, 1.6902785321094298], [2.3383660289936534, 2.7024286789292775, 1.6902785321094298], [2.1313733586597943, 3.0023093334202904, 1.6902785321094298], [1.8586293810954611, 3.2439394126991057, 1.6902785321094298], [1.5359849848926723, 3.4132762548476405, 1.6902785321094298], [1.1821910895940713, 3.5004786181145988, 1.6902785321094298], [0.81780891040592818, 3.5004786181145988, 1.6902785321094298], [0.46401501510732768, 3.4132762548476405, 1.6902785321094298], [0.14137061890453906, 3.2439394126991057, 1.6902785321094298], [-0.13137335865979427, 3.0023093334202904, 1.6902785321094298], [-0.33836602899365342, 2.7024286789292771, 1.6902785321094298], [-0.46757773048497753, 2.3617254228432567, 1.6902785321094298], [-0.51149914870851654, 2.0, 1.6902785321094298], [-0.46757773048497731, 1.6382745771567424, 1.6902785321094298], [-0.33836602899365276, 1.297571321070722, 1.6902785321094298], [-0.13137335865979383, 0.9976906665797094, 1.6902785321094298], [0.14137061890453873, 0.75606058730089454, 1.6902785321094298], [0.46401501510732734, 0.58672374515235992, 1.6902785321094298], [0.81780891040592907, 0.49952138188540141, 1.6902785321094298], [1.1821910895940717, 0.49952138188540141, 1.6902785321094298], [1.5359849848926723, 0.5867237451523597, 1.6902785321094298], [1.8586293810954617, 0.75606058730089498, 1.6902785321094298], [2.1313733586597943, 0.99769066657971006, 1.6902785321094298], [2.3383660289936534, 1.2975713210707225, 1.6902785321094298], [2.4675777304849777, 1.6382745771567433, 1.6902785321094298], [2.5114991487085163, 1.9999999999999996, 1.6902785321094298], [2.682507065662362, 2.0, 1.9187183650888047], [2.6336164681663878, 2.402650296087498, 1.9187183650888047], [2.4897860194948396, 2.7819000205406663, 1.9187183650888047], [2.2593746225220985, 3.1157085578909456, 1.9187183650888047], [1.9557729501288699, 3.3846761692922032, 1.9187183650888047], [1.5966252279674698, 3.5731714348272847, 1.9187183650888047], [1.2028038162010659, 3.6702396948157041, 1.9187183650888047], [0.79719618379893353, 3.6702396948157041, 1.9187183650888047], [0.40337477203252992, 3.5731714348272847, 1.9187183650888047], [0.044227049871130242, 3.3846761692922032, 1.9187183650888047], [-0.25937462252209875, 3.1157085578909456, 1.9187183650888047], [-0.4897860194948398, 2.7819000205406654, 1.9187183650888047], [-0.63361646816638784, 2.4026502960874976, 1.9187183650888047], [-0.68250706566236219, 2.0, 1.9187183650888047], [-0.63361646816638761, 1.5973497039125013, 1.9187183650888047], [-0.48978601949483913, 1.2180999794593332, 1.9187183650888047], [-0.25937462252209853, 0.88429144210905397, 1.9187183650888047], [0.044227049871129909, 0.61532383070779728, 1.9187183650888047], [0.40337477203252947, 0.42682856517271572, 1.9187183650888047], [0.79719618379893464, 0.32976030518429567, 1.9187183650888047], [1.2028038162010661, 0.32976030518429589, 1.9187183650888047], [1.5966252279674698, 0.4268285651727155, 1.9187183650888047], [1.9557729501288708, 0.61532383070779773, 1.9187183650888047], [2.259374622522099, 0.88429144210905486, 1.9187183650888047], [2.48978601949484, 1.2180999794593341, 1.9187183650888047], [2.6336164681663878, 1.5973497039125022, 1.9187183650888047], [2.682507065662362, 1.9999999999999996, 1.9187183650888047], [2.8192639907090369, 2.0, 2.1691699739962269], [2.7663994855168044, 2.4353783704509664, 2.1691699739962269], [2.6108782628272218, 2.8454541325473084, 2.1691699739962269], [2.3617386508063642, 3.2063951735607334, 2.1691699739962269], [2.033459738119241, 3.4972249121548442, 2.1691699739962269], [1.6451199019259304, 3.7010413810456368, 2.1691699739962269], [1.219288041948118, 3.8059995079039002, 2.1691699739962269], [0.78071195805188132, 3.8059995079039002, 2.1691699739962269], [0.35488009807406939, 3.7010413810456368, 2.1691699739962269], [-0.033459738119240523, 3.4972249121548442, 2.1691699739962269], [-0.36173865080636425, 3.2063951735607334, 2.1691699739962269], [-0.61087826282722202, 2.8454541325473079, 2.1691699739962269], [-0.76639948551680415, 2.4353783704509664, 2.1691699739962269], [-0.81926399070903666, 2.0000000000000004, 2.1691699739962269], [-0.76639948551680415, 1.5646216295490327, 2.1691699739962269], [-0.61087826282722135, 1.1545458674526907, 2.1691699739962269], [-0.3617386508063638, 0.79360482643926611, 2.1691699739962269], [-0.033459738119240967, 0.50277508784515601, 2.1691699739962269], [0.35488009807406895, 0.29895861895436338, 2.1691699739962269], [0.78071195805188254, 0.1940004920960996, 2.1691699739962269], [1.2192880419481185, 0.1940004920960996, 2.1691699739962269], [1.6451199019259304, 0.29895861895436315, 2.1691699739962269], [2.0334597381192414, 0.50277508784515668, 2.1691699739962269], [2.3617386508063647, 0.793604826439267, 2.1691699739962269], [2.6108782628272218, 1.1545458674526916, 2.1691699739962269], [2.7663994855168044, 1.5646216295490336, 2.1691699739962269], [2.8192639907090369, 1.9999999999999996, 2.1691699739962269], [2.918985947228995, 2.0, 2.4365348863171405], [2.863223703217574, 2.4592433967195952, 2.4365348863171405], [2.6991776701177459, 2.8917972365036744, 2.4365348863171405], [2.4363816070902038, 3.2725230624532213, 2.4365348863171405], [2.0901082660932859, 3.5792944734461183, 2.4365348863171405], [1.6804817950533508, 3.7942830301441663, 2.4365348863171405], [1.2313081955355996, 3.904994379083683, 2.4365348863171405], [0.76869180446439989, 3.904994379083683, 2.4365348863171405], [0.31951820494664895, 3.7942830301441663, 2.4365348863171405], [-0.090108266093285705, 3.5792944734461183, 2.4365348863171405], [-0.43638160709020424, 3.2725230624532213, 2.4365348863171405], [-0.69917767011774634, 2.8917972365036739, 2.4365348863171405], [-0.86322370321757402, 2.4592433967195948, 2.4365348863171405], [-0.91898594722899474, 2.0000000000000004, 2.4365348863171405], [-0.8632237032175738, 1.5407566032804039, 2.4365348863171405], [-0.69917767011774568, 1.1082027634963252, 2.4365348863171405], [-0.43638160709020379, 0.72747693754677845, 2.4365348863171405], [-0.090108266093286149, 0.42070552655388194, 2.4365348863171405], [0.31951820494664862, 0.20571696985583388, 2.4365348863171405], [0.76869180446440111, 0.095005620916316813, 2.4365348863171405], [1.2313081955355998, 0.095005620916317035, 2.4365348863171405], [1.6804817950533508, 0.20571696985583365, 2.4365348863171405], [2.0901082660932868, 0.4207055265538826, 2.4365348863171405], [2.4363816070902047, 0.72747693754677933, 2.4365348863171405], [2.6991776701177463, 1.1082027634963261, 2.4365348863171405], [2.863223703217574, 1.5407566032804048, 2.4365348863171405], [2.918985947228995, 1.9999999999999996, 2.4365348863171405], [2.9796428837618656, 2.0, 2.7153703234534299], [2.9221180594142959, 2.4737595517796072, 2.7153703234534299], [2.7528867200684406, 2.9199859204556873, 2.7153703234534299], [2.4817839760361897, 3.3127460514476414, 2.7153703234534299], [2.1245653333823187, 3.6292141535672062, 2.7153703234534299], [1.7019910411809356, 3.8509982510339387, 2.7153703234534299], [1.2386195812997296, 3.965209058255466, 2.7153703234534299], [0.76138041870026985, 3.965209058255466, 2.7153703234534299], [0.2980089588190642, 3.8509982510339387, 2.7153703234534299], [-0.12456533338231868, 3.6292141535672062, 2.7153703234534299], [-0.48178397603619016, 3.3127460514476414, 2.7153703234534299], [-0.75288672006844082, 2.9199859204556868, 2.7153703234534299], [-0.92211805941429614, 2.4737595517796072, 2.7153703234534299], [-0.97964288376186537, 2.0000000000000004, 2.7153703234534299], [-0.92211805941429592, 1.5262404482203915, 2.7153703234534299], [-0.75288672006844037, 1.0800140795443118, 2.7153703234534299], [-0.48178397603618972, 0.68725394855235811, 2.7153703234534299], [-0.12456533338231912, 0.37078584643279378, 2.7153703234534299], [0.29800895881906375, 0.14900174896606155, 2.7153703234534299], [0.76138041870027107, 0.034790941744533788, 2.7153703234534299], [1.2386195812997298, 0.034790941744533788, 2.7153703234534299], [1.7019910411809356, 0.14900174896606133, 2.7153703234534299], [2.1245653333823196, 0.37078584643279444, 2.7153703234534299], [2.4817839760361906, 0.687253948552359, 2.7153703234534299], [2.7528867200684406, 1.0800140795443127, 2.7153703234534299], [2.9221180594142959, 1.5262404482203926, 2.7153703234534299], [2.9796428837618656, 1.9999999999999996, 2.7153703234534299], [3.0, 2.0, 3.0], [2.941883634852104, 2.4786313285751156, 3.0], [2.7709120513064196, 2.9294463440875371, 3.0], [2.4970214963422022, 3.3262453164815904, 3.0], [2.1361294934623114, 3.6459677317873131, 3.0], [1.7092097740850711, 3.8700324853708299, 3.0], [1.2410733605106461, 3.9854177481961077, 3.0], [0.75892663948935335, 3.9854177481961077, 3.0], [0.29079022591492865, 3.8700324853708299, 3.0], [-0.1361294934623114, 3.6459677317873131, 3.0], [-0.49702149634220238, 3.3262453164815904, 3.0], [-0.77091205130642004, 2.9294463440875367, 3.0], [-0.94188363485210402, 2.4786313285751156, 3.0], [-1.0, 2.0000000000000004, 3.0], [-0.9418836348521038, 1.5213686714248835, 3.0], [-0.77091205130641938, 1.0705536559124624, 3.0], [-0.49702149634220194, 0.67375468351840939, 3.0], [-0.13612949346231185, 0.3540322682126873, 3.0], [0.2907902259149282, 0.12996751462917056, 3.0], [0.75892663948935457, 0.014582251803891833, 3.0], [1.2410733605106463, 0.014582251803892055, 3.0], [1.7092097740850711, 0.12996751462917033, 3.0], [2.1361294934623123, 0.35403226821268796, 3.0], [2.4970214963422031, 0.67375468351841028, 3.0], [2.77091205130642, 1.0705536559124633, 3.0], [2.941883634852104, 1.5213686714248844, 3.0], [3.0, 1.9999999999999996, 3.0], [2.9796428837618656, 2.0, 3.2846296765465701], [2.9221180594142964, 2.4737595517796072, 3.2846296765465701], [2.7528867200684406, 2.9199859204556877, 3.2846296765465701], [2.4817839760361902, 3.3127460514476419, 3.2846296765465701], [2.1245653333823191, 3.6292141535672062, 3.2846296765465701], [1.7019910411809356, 3.8509982510339391, 3.2846296765465701], [1.2386195812997296, 3.9652090582554664, 3.2846296765465701], [0.76138041870026985, 3.9652090582554664, 3.2846296765465701], [0.29800895881906408, 3.8509982510339391, 3.2846296765465701], [-0.1245653333823189, 3.6292141535672062, 3.2846296765465701], [-0.48178397603619039, 3.3127460514476419, 3.2846296765465701], [-0.75288672006844104, 2.9199859204556873, 3.2846296765465701], [-0.92211805941429636, 2.4737595517796072, 3.2846296765465701], [-0.97964288376186559, 2.0000000000000004, 3.2846296765465701], [-0.92211805941429614, 1.5262404482203915, 3.2846296765465701], [-0.75288672006844037, 1.0800140795443118, 3.2846296765465701], [-0.48178397603618994, 0.68725394855235789, 3.2846296765465701], [-0.12456533338231934, 0.37078584643279378, 3.2846296765465701], [0.29800895881906364, 0.14900174896606133, 3.2846296765465701], [0.76138041870027107, 0.034790941744533566, 3.2846296765465701], [1.23861958129973, 0.034790941744533566, 3.2846296765465701], [1.7019910411809356, 0.14900174896606111, 3.2846296765465701], [2.12456533338232, 0.37078584643279444, 3.2846296765465701], [2.4817839760361906, 0.68725394855235877, 3.2846296765465701], [2.752886720068441, 1.0800140795443127, 3.2846296765465701], [2.9221180594142964, 1.5262404482203926, 3.2846296765465701], [2.9796428837618656, 1.9999999999999996, 3.2846296765465701], [2.918985947228995, 2.0, 3.5634651136828595], [2.863223703217574, 2.4592433967195952, 3.5634651136828595], [2.6991776701177459, 2.8917972365036744, 3.5634651136828595], [2.4363816070902038, 3.2725230624532213, 3.5634651136828595], [2.0901082660932859, 3.5792944734461183, 3.5634651136828595], [1.6804817950533508, 3.7942830301441663, 3.5634651136828595], [1.2313081955355996, 3.904994379083683, 3.5634651136828595], [0.76869180446439989, 3.904994379083683, 3.5634651136828595], [0.31951820494664895, 3.7942830301441663, 3.5634651136828595], [-0.090108266093285705, 3.5792944734461183, 3.5634651136828595], [-0.43638160709020424, 3.2725230624532213, 3.5634651136828595], [-0.69917767011774634, 2.8917972365036739, 3.5634651136828595], [-0.86322370321757402, 2.4592433967195948, 3.5634651136828595], [-0.91898594722899474, 2.0000000000000004, 3.5634651136828595], [-0.8632237032175738, 1.5407566032804039, 3.5634651136828595], [-0.69917767011774568, 1.1082027634963252, 3.5634651136828595], [-0.43638160709020379, 0.72747693754677845, 3.5634651136828595], [-0.090108266093286149, 0.42070552655388194, 3.5634651136828595], [0.31951820494664862, 0.20571696985583388, 3.5634651136828595], [0.76869180446440111, 0.095005620916316813, 3.5634651136828595], [1.2313081955355998, 0.095005620916317035, 3.5634651136828595], [1.6804817950533508, 0.20571696985583365, 3.5634651136828595], [2.0901082660932868, 0.4207055265538826, 3.5634651136828595], [2.4363816070902047, 0.72747693754677933, 3.5634651136828595], [2.6991776701177463, 1.1082027634963261, 3.5634651136828595], [2.863223703217574, 1.5407566032804048, 3.5634651136828595], [2.918985947228995, 1.9999999999999996, 3.5634651136828595], [2.8192639907090369, 2.0, 3.8308300260037726], [2.7663994855168044, 2.4353783704509664, 3.8308300260037726], [2.6108782628272218, 2.8454541325473088, 3.8308300260037726], [2.3617386508063642, 3.2063951735607339, 3.8308300260037726], [2.033459738119241, 3.4972249121548442, 3.8308300260037726], [1.6451199019259306, 3.7010413810456368, 3.8308300260037726], [1.219288041948118, 3.8059995079039006, 3.8308300260037726], [0.78071195805188132, 3.8059995079039006, 3.8308300260037726], [0.35488009807406928, 3.7010413810456368, 3.8308300260037726], [-0.033459738119240745, 3.4972249121548442, 3.8308300260037726], [-0.36173865080636447, 3.2063951735607339, 3.8308300260037726], [-0.61087826282722224, 2.8454541325473084, 3.8308300260037726], [-0.76639948551680437, 2.4353783704509664, 3.8308300260037726], [-0.81926399070903688, 2.0000000000000004, 3.8308300260037726], [-0.76639948551680437, 1.5646216295490325, 3.8308300260037726], [-0.61087826282722157, 1.1545458674526907, 3.8308300260037726], [-0.36173865080636403, 0.79360482643926589, 3.8308300260037726], [-0.033459738119240967, 0.50277508784515601, 3.8308300260037726], [0.35488009807406895, 0.29895861895436315, 3.8308300260037726], [0.78071195805188243, 0.19400049209609938, 3.8308300260037726], [1.2192880419481185, 0.19400049209609938, 3.8308300260037726], [1.6451199019259306, 0.29895861895436293, 3.8308300260037726], [2.0334597381192419, 0.50277508784515645, 3.8308300260037726], [2.3617386508063651, 0.79360482643926678, 3.8308300260037726], [2.6108782628272222, 1.1545458674526916, 3.8308300260037726], [2.7663994855168044, 1.5646216295490334, 3.8308300260037726], [2.8192639907090369, 1.9999999999999996, 3.8308300260037726], [2.682507065662362, 2.0, 4.0812816349111953], [2.6336164681663878, 2.402650296087498, 4.0812816349111953], [2.4897860194948396, 2.7819000205406663, 4.0812816349111953], [2.2593746225220985, 3.1157085578909456, 4.0812816349111953], [1.9557729501288699, 3.3846761692922032, 4.0812816349111953], [1.5966252279674698, 3.5731714348272847, 4.0812816349111953], [1.2028038162010659, 3.6702396948157041, 4.0812816349111953], [0.79719618379893353, 3.6702396948157041, 4.0812816349111953], [0.40337477203252992, 3.5731714348272847, 4.0812816349111953], [0.044227049871130242, 3.3846761692922032, 4.0812816349111953], [-0.25937462252209875, 3.1157085578909456, 4.0812816349111953], [-0.4897860194948398, 2.7819000205406654, 4.0812816349111953], [-0.63361646816638784, 2.4026502960874976, 4.0812816349111953], [-0.68250706566236219, 2.0, 4.0812816349111953], [-0.63361646816638761, 1.5973497039125013, 4.0812816349111953], [-0.48978601949483913, 1.2180999794593332, 4.0812816349111953], [-0.25937462252209853, 0.88429144210905397, 4.0812816349111953], [0.044227049871129909, 0.61532383070779728, 4.0812816349111953], [0.40337477203252947, 0.42682856517271572, 4.0812816349111953], [0.79719618379893464, 0.32976030518429567, 4.0812816349111953], [1.2028038162010661, 0.32976030518429589, 4.0812816349111953], [1.5966252279674698, 0.4268285651727155, 4.0812816349111953], [1.9557729501288708, 0.61532383070779773, 4.0812816349111953], [2.259374622522099, 0.88429144210905486, 4.0812816349111953], [2.48978601949484, 1.2180999794593341, 4.0812816349111953], [2.6336164681663878, 1.5973497039125022, 4.0812816349111953], [2.682507065662362, 1.9999999999999996, 4.0812816349111953], [2.5114991487085163, 2.0, 4.3097214678905704], [2.4675777304849777, 2.3617254228432567, 4.3097214678905704], [2.3383660289936534, 2.7024286789292775, 4.3097214678905704], [2.1313733586597943, 3.0023093334202904, 4.3097214678905704], [1.8586293810954611, 3.2439394126991057, 4.3097214678905704], [1.5359849848926723, 3.4132762548476405, 4.3097214678905704], [1.1821910895940713, 3.5004786181145988, 4.3097214678905704], [0.81780891040592818, 3.5004786181145988, 4.3097214678905704], [0.46401501510732768, 3.4132762548476405, 4.3097214678905704], [0.14137061890453906, 3.2439394126991057, 4.3097214678905704], [-0.13137335865979427, 3.0023093334202904, 4.3097214678905704], [-0.33836602899365342, 2.7024286789292771, 4.3097214678905704], [-0.46757773048497753, 2.3617254228432567, 4.3097214678905704], [-0.51149914870851654, 2.0, 4.3097214678905704], [-0.46757773048497731, 1.6382745771567424, 4.3097214678905704], [-0.33836602899365276, 1.297571321070722, 4.3097214678905704], [-0.13137335865979383, 0.9976906665797094, 4.3097214678905704], [0.14137061890453873, 0.75606058730089454, 4.3097214678905704], [0.46401501510732734, 0.58672374515235992, 4.3097214678905704], [0.81780891040592907, 0.49952138188540141, 4.3097214678905704], [1.1821910895940717, 0.49952138188540141, 4.3097214678905704], [1.5359849848926723, 0.5867237451523597, 4.3097214678905704], [1.8586293810954617, 0.75606058730089498, 4.3097214678905704], [2.1313733586597943, 0.99769066657971006, 4.3097214678905704], [2.3383660289936534, 1.2975713210707225, 4.3097214678905704], [2.4675777304849777, 1.6382745771567433, 4.3097214678905704], [2.5114991487085163, 1.9999999999999996, 4.3097214678905704], [2.3097214678905704, 2.0, 4.5114991487085163], [2.2716633423555868, 2.3134368631199069, 4.5114991487085163], [2.1597007656710727, 2.6086579150519267, 4.5114991487085163], [1.9803405958265237, 2.8685059813426315, 4.5114991487085163], [1.7440065939456146, 3.077879636888496, 4.5114991487085163], [1.4644336331785195, 3.2246108458714673, 4.5114991487085163], [1.157869477798658, 3.3001721237716986, 4.5114991487085163], [0.84213052220134155, 3.3001721237716986, 4.5114991487085163], [0.53556636682148029, 3.2246108458714673, 4.5114991487085163], [0.25599340605438559, 3.077879636888496, 4.5114991487085163], [0.019659404173476225, 2.8685059813426315, 4.5114991487085163], [-0.15970076567107272, 2.6086579150519262, 4.5114991487085163], [-0.27166334235558698, 2.3134368631199069, 4.5114991487085163], [-0.30972146789057042, 2.0, 4.5114991487085163], [-0.27166334235558698, 1.6865631368800922, 4.5114991487085163], [-0.15970076567107228, 1.3913420849480729, 4.5114991487085163], [0.019659404173476558, 1.1314940186573685, 4.5114991487085163], [0.25599340605438536, 0.92212036311150403, 4.5114991487085163], [0.53556636682147996, 0.77538915412853271, 4.5114991487085163], [0.84213052220134244, 0.69982787622830123, 4.5114991487085163], [1.1578694777986582, 0.69982787622830123, 4.5114991487085163], [1.4644336331785195, 0.77538915412853271, 4.5114991487085163], [1.7440065939456151, 0.92212036311150447, 4.5114991487085163], [1.9803405958265241, 1.131494018657369, 4.5114991487085163], [2.1597007656710727, 1.3913420849480735, 4.5114991487085163], [2.2716633423555868, 1.6865631368800929, 4.5114991487085163], [2.3097214678905704, 1.9999999999999998, 4.5114991487085163], [2.0812816349111949, 2.0, 4.6825070656623629], [2.0498615557500885, 2.2587676327407094, 4.6825070656623629], [1.9574273390602719, 2.5024966312486026, 4.6825070656623629], [1.8093509255310498, 2.7170223520492645, 4.6825070656623629], [1.614237978080878, 2.8898773400190283, 4.6825070656623629], [1.3834277520088525, 3.0110158915594081, 4.6825070656623629], [1.1303340986932435, 3.0733978743755954, 4.6825070656623629], [0.86966590130675603, 3.0733978743755954, 4.6825070656623629], [0.61657224799114752, 3.0110158915594081, 4.6825070656623629], [0.38576202191912212, 2.8898773400190283, 4.6825070656623629], [0.19064907446895007, 2.7170223520492645, 4.6825070656623629], [0.042572660939728113, 2.5024966312486021, 4.6825070656623629], [-0.049861555750088327, 2.2587676327407089, 4.6825070656623629], [-0.081281634911194889, 2.0, 4.6825070656623629], [-0.049861555750088327, 1.7412323672592902, 4.6825070656623629], [0.042572660939728557, 1.497503368751397, 4.6825070656623629], [0.1906490744689503, 1.2829776479507353, 4.6825070656623629], [0.3857620219191219, 1.1101226599809717, 4.6825070656623629], [0.61657224799114718, 0.98898410844059215, 4.6825070656623629], [0.86966590130675669, 0.92660212562440458, 4.6825070656623629], [1.1303340986932437, 0.92660212562440458, 4.6825070656623629], [1.3834277520088525, 0.98898410844059215, 4.6825070656623629], [1.6142379780808787, 1.1101226599809721, 4.6825070656623629], [1.8093509255310503, 1.2829776479507358, 4.6825070656623629], [1.9574273390602719, 1.4975033687513974, 4.6825070656623629], [2.0498615557500885, 1.7412323672592909, 4.6825070656623629], [2.0812816349111949, 1.9999999999999998, 4.6825070656623629], [1.8308300260037726, 2.0, 4.8192639907090369], [1.8066876154202371, 2.1988306395831416, 4.8192639907090369], [1.7356634528186534, 2.3861059651136798, 4.8192639907090369], [1.6218852043670993, 2.5509422153898909, 4.8192639907090369], [1.4719652482984726, 2.6837597067011121, 4.8192639907090369], [1.2946163875226147, 2.7768395692242729, 4.8192639907090369], [1.1001454931909385, 2.8247723396810622, 4.8192639907090369], [0.89985450680906132, 2.8247723396810622, 4.8192639907090369], [0.70538361247738524, 2.7768395692242729, 4.8192639907090369], [0.52803475170152736, 2.6837597067011121, 4.8192639907090369], [0.37811479563290062, 2.5509422153898909, 4.8192639907090369], [0.26433654718134636, 2.3861059651136798, 4.8192639907090369], [0.19331238457976296, 2.1988306395831416, 4.8192639907090369], [0.16916997399622735, 2.0, 4.8192639907090369], [0.19331238457976307, 1.8011693604168579, 4.8192639907090369], [0.26433654718134658, 1.6138940348863198, 4.8192639907090369], [0.37811479563290085, 1.4490577846101091, 4.8192639907090369], [0.52803475170152714, 1.3162402932988881, 4.8192639907090369], [0.70538361247738501, 1.2231604307757271, 4.8192639907090369], [0.89985450680906187, 1.1752276603189378, 4.8192639907090369], [1.1001454931909387, 1.175227660318938, 4.8192639907090369], [1.2946163875226147, 1.2231604307757271, 4.8192639907090369], [1.4719652482984731, 1.3162402932988884, 4.8192639907090369], [1.6218852043670995, 1.4490577846101096, 4.8192639907090369], [1.7356634528186536, 1.6138940348863202, 4.8192639907090369], [1.8066876154202371, 1.8011693604168582, 4.8192639907090369], [1.8308300260037726, 1.9999999999999998, 4.8192639907090369], [1.5634651136828595, 2.0, 4.918985947228995], [1.5470918415354125, 2.1348460279838779, 4.918985947228995], [1.4989235801558587, 2.261855294966701, 4.918985947228995], [1.4217596938110715, 2.3736464840113296, 4.918985947228995], [1.3200846670960953, 2.4637226975549282, 4.918985947228995], [1.1998074829899199, 2.5268490334800573, 4.918985947228995], [1.0679182142430201, 2.5593568185976432, 4.918985947228995], [0.93208178575697975, 2.5593568185976432, 4.918985947228995], [0.8001925170100801, 2.5268490334800573, 4.918985947228995], [0.67991533290390471, 2.4637226975549282, 4.918985947228995], [0.5782403061889283, 2.3736464840113296, 4.918985947228995], [0.5010764198441412, 2.261855294966701, 4.918985947228995], [0.45290815846458754, 2.1348460279838779, 4.918985947228995], [0.43653488631714066, 2.0, 4.918985947228995], [0.45290815846458765, 1.8651539720161219, 4.918985947228995], [0.50107641984414131, 1.7381447050332988, 4.918985947228995], [0.57824030618892852, 1.6263535159886704, 4.918985947228995], [0.67991533290390449, 1.5362773024450718, 4.918985947228995], [0.80019251701007998, 1.4731509665199427, 4.918985947228995], [0.93208178575698009, 1.4406431814023568, 4.918985947228995], [1.0679182142430201, 1.4406431814023568, 4.918985947228995], [1.1998074829899199, 1.4731509665199427, 4.918985947228995], [1.3200846670960957, 1.536277302445072, 4.918985947228995], [1.4217596938110717, 1.6263535159886706, 4.918985947228995], [1.4989235801558589, 1.738144705033299, 4.918985947228995], [1.5470918415354125, 1.8651539720161223, 4.918985947228995], [1.5634651136828595, 1.9999999999999998, 4.918985947228995], [1.2846296765465703, 2.0, 4.9796428837618656], [1.2763588554395162, 2.0681163401186953, 4.9796428837618656], [1.2520270621778848, 2.1322740061425138, 4.9796428837618656], [1.2130483721435719, 2.1887443877257793, 4.9796428837618656], [1.1616880851195983, 2.2342456315523576, 4.9796428837618656], [1.1009310743007501, 2.2661333707213389, 4.9796428837618656], [1.03430831631307, 2.2825544057394391, 4.9796428837618656], [0.96569168368692992, 2.2825544057394391, 4.9796428837618656], [0.8990689256992499, 2.2661333707213389, 4.9796428837618656], [0.83831191488040169, 2.2342456315523576, 4.9796428837618656], [0.78695162785642814, 2.1887443877257793, 4.9796428837618656], [0.74797293782211516, 2.1322740061425138, 4.9796428837618656], [0.72364114456048367, 2.0681163401186953, 4.9796428837618656], [0.71537032345342966, 2.0, 4.9796428837618656], [0.72364114456048367, 1.9318836598813047, 4.9796428837618656], [0.74797293782211516, 1.8677259938574859, 4.9796428837618656], [0.78695162785642814, 1.8112556122742205, 4.9796428837618656], [0.83831191488040169, 1.7657543684476427, 4.9796428837618656], [0.8990689256992499, 1.7338666292786609, 4.9796428837618656], [0.96569168368693015, 1.7174455942605609, 4.9796428837618656], [1.03430831631307, 1.7174455942605609, 4.9796428837618656], [1.1009310743007501, 1.7338666292786609, 4.9796428837618656], [1.1616880851195983, 1.7657543684476427, 4.9796428837618656], [1.2130483721435721, 1.8112556122742207, 4.9796428837618656], [1.2520270621778848, 1.8677259938574859, 4.9796428837618656], [1.2763588554395162, 1.9318836598813047, 4.9796428837618656], [1.2846296765465703, 2.0, 4.9796428837618656], [1.0000000000000002, 2.0, 5.0], [1.0000000000000002, 2.0, 5.0], [1.0000000000000002, 2.0, 5.0], [1.0000000000000002, 2.0, 5.0], [1.0000000000000002, 2.0, 5.0], [1.0, 2.0000000000000004, 5.0], [1.0, 2.0000000000000004, 5.0], [1.0, 2.0000000000000004, 5.0], [0.99999999999999989, 2.0000000000000004, 5.0], [0.99999999999999989, 2.0, 5.0], [0.99999999999999978, 2.0, 5.0], [0.99999999999999978, 2.0, 5.0], [0.99999999999999978, 2.0, 5.0], [0.99999999999999978, 2.0, 5.0], [0.99999999999999978, 2.0, 5.0], [0.99999999999999978, 1.9999999999999998, 5.0], [0.99999999999999978, 1.9999999999999998, 5.0], [0.99999999999999989, 1.9999999999999998, 5.0], [0.99999999999999989, 1.9999999999999998, 5.0], [1.0, 1.9999999999999998, 5.0], [1.0, 1.9999999999999998, 5.0], [1.0, 1.9999999999999998, 5.0], [1.0000000000000002, 1.9999999999999998, 5.0], [1.0000000000000002, 1.9999999999999998, 5.0], [1.0000000000000002, 1.9999999999999998, 5.0], [1.0000000000000002, 2.0, 5.0], [1.0000000000000002, 2.0, 5.0]])

if __name__ == "__main__":
    unittest.main()
