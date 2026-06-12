/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-11-28 08:29:33
 * @LastEditTime: 2024-01-23 03:44:33
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /LineRepresentation/src/utils/UtilParameterss.hpp
 */
/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-11-28 08:29:33
 * @LastEditTime: 2024-01-19 03:00:56
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /LineRepresentation/src/utils/UtilParameterss.hpp
 */

#pragma once

// paraline1
const int line_size = 53;
const int global_size_vec = 3 * line_size + 3 /*3*n+3*/;
const int local_size_vec = 2 * line_size + 2 /*2n+2*/;
const int unique_size_vec = 3 * line_size + 1 /*3n+1*/;
const int para_size = line_size + 1;
const int jacobian_pose_lines_parameters = 2 * line_size + 2 + 6;

// paraline2
const int line_size_n2 = 83;                         // 86
const int global_size_vec_16 = 3 * line_size_n2 + 3; /*3*15+3*/
const int local_size_vec_16 = 2 * line_size_n2 + 2;  /*2*15+2*/
const int unique_size_vec_16 = 3 * line_size_n2 + 1 /*3n+1*/;
const int para_size_16 = line_size_n2 + 1; /*16+1*/
const int jacobian_pose_lines_parameters_16 = 2 * line_size_n2 + 2 + 6;

// paraline3
const int line_size_n3 = 26;
const int global_size_vec_n3 = 3 * line_size_n3 + 3; /*3*15+3*/
const int local_size_vec_n3 = 2 * line_size_n3 + 2;  /*2*15+2*/
const int unique_size_vec_n3 = 3 * line_size_n3 + 1 /*3n+1*/;
const int para_size_n3 = line_size_n3 + 1; /*n3+1*/
const int jacobian_pose_lines_parameters_n3 = 2 * line_size_n3 + 2 + 6;