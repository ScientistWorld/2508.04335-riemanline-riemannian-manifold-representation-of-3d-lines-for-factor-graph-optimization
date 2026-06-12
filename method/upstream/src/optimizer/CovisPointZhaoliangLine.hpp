#ifndef __VENOM_SRC_OPTIMIZER_COVISZHAO_BUNDLEADJUSTMENT__
#define __VENOM_SRC_OPTIMIZER_COVISZHAO_BUNDLEADJUSTMENT__

#include <ceres/ceres.h>

#include "src/manager_env/EnvTrajectory.hpp"
#include "src/optimizer/factor/LineProjectionFactor.hpp"
#include "src/optimizer/factor/RiemanLineProjectionFactor.hpp"
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/optimizer/parametrization/pose_local_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"

namespace simulator
{
    namespace optimizer
    {


        class CovisPointZhaoliangLineBA
        {
        public:
            static void optimizer(
                simulator::Trajectory *ptr_robot_trajectory,
                simulator::OptimizationManager &opti_para,
                std::map<int, Eigen::Vector3d> &optimized_mappoints,
                std::map<int, Eigen::Matrix<double, 3, 2>> &optimized_maplines,
                std::map<int, Eigen::Matrix4d> &optimized_poses)
            {
                // set parameters
                constexpr int min_obser_num = 3;

                // intrinsic
                double fx, fy, cx, cy;

                fx = ptr_robot_trajectory->cam_intri.fx;
                fy = ptr_robot_trajectory->cam_intri.fy;
                cx = ptr_robot_trajectory->cam_intri.cx;
                cy = ptr_robot_trajectory->cam_intri.cy;

                // construct ceres optim parameters
                ceres::Problem problem;
                ceres::LossFunction *loss_function;
                ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering();
                loss_function = new ceres::CauchyLoss(1.0);

                // add mappoints
                // double Para_Point_Feature[opti_para.mappoints.size()][3];
                auto iteration_pt = opti_para.mappoints.rbegin();
                double Para_Point_Feature[iteration_pt->first + 1][3];
                for (auto mp = opti_para.mappoints.begin(); mp != opti_para.mappoints.end(); mp++)
                {
                    int mp_id = mp->first;
                    if (opti_para.mappoints[mp_id] == Vec3::Zero())
                        continue;
                    Para_Point_Feature[mp_id][0] = mp->second(0);
                    Para_Point_Feature[mp_id][1] = mp->second(1);
                    Para_Point_Feature[mp_id][2] = mp->second(2);
                    problem.AddParameterBlock(Para_Point_Feature[mp_id], 3);
                }

                // add maplines
                // double Para_Line_Feature[opti_para.maplines.size()][4];
                auto iteration_ml = opti_para.maplines.rbegin();
                double Para_Line_Feature[iteration_ml->first + 1][4];
                double Para_Line_Feature_Host_id[iteration_ml->first + 1][2];
                for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                {
                    int ml_id = asso_ml_mea->first;
                    Para_Line_Feature_Host_id[ml_id][0] = 0;
                    Para_Line_Feature_Host_id[ml_id][1] = 0;

                    if (opti_para.maplines[ml_id] == Mat32::Zero())
                        assert(1 == 0);
                    if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;

                    // get host kf id0 id1 by norm intersect angle
                    Para_Line_Feature_Host_id[ml_id][0] = asso_ml_mea->second[0].first;
                    Eigen::Vector4d ml_obser_0 = asso_ml_mea->second[0].second;
                    Eigen::Vector3d host0_norm = Eigen::Vector3d((ml_obser_0(0) - cx) / fx, (ml_obser_0(1) - cy) / fy, 1).cross(Eigen::Vector3d((ml_obser_0(2) - cx) / fx, (ml_obser_0(3) - cy) / fy, 1));
                    host0_norm = host0_norm.normalized();
                    Eigen::Vector3d hosti_norm_maxangle;
                    Eigen::Matrix4d Twc0 = opti_para.kfs[asso_ml_mea->second[0].first];
                    Eigen::Matrix3d Rot0 = Twc0.block(0, 0, 3, 3);
                    double max_angle = 0;
                    for (int i = 1; i < asso_ml_mea->second.size(); i++)
                    {
                        int kf_id_i = asso_ml_mea->second[i].first;
                        Eigen::Vector4d ml_obser_i = asso_ml_mea->second[i].second;

                        Eigen::Vector4d mp_obser_i_unit(
                            (ml_obser_i(0) - cx) / fx,
                            (ml_obser_i(1) - cy) / fy,
                            (ml_obser_i(2) - cx) / fx,
                            (ml_obser_i(3) - cy) / fy);
                        
                        Eigen::Vector3d hosti_norm = Eigen::Vector3d((ml_obser_i(0) - cx) / fx, (ml_obser_i(1) - cy) / fy, 1).cross(Eigen::Vector3d((ml_obser_i(2) - cx) / fx, (ml_obser_i(3) - cy) / fy, 1));
                        hosti_norm = hosti_norm.normalized();

                        Eigen::Matrix4d Twci = opti_para.kfs[kf_id_i];
                        Eigen::Matrix3d Roti = Twci.block(0, 0, 3, 3);

                        Eigen::Vector3d host0_w = Rot0 * host0_norm;
                        Eigen::Vector3d hosti_w = Roti * hosti_norm;

                        double cur_angle = abs(acos(host0_w.transpose() * hosti_w));

                        if(cur_angle > max_angle)
                        {
                            max_angle = cur_angle;
                            Para_Line_Feature_Host_id[ml_id][1] = kf_id_i;
                            hosti_norm_maxangle = hosti_norm;
                        }
                    }

                    if(max_angle < 0.0001)
                        continue;

                    // add zhaoliang line parameterization, using 2 angle present a unit vector
                    {
                        // transform norm vector from camera frame to world frame
                        host0_norm = Rot0 * host0_norm;

                        Eigen::Matrix4d Twc1 = opti_para.kfs[Para_Line_Feature_Host_id[ml_id][1]];
                        Eigen::Matrix3d Rot1 = Twc1.block(0, 0, 3, 3);
                        hosti_norm_maxangle = Rot1 * hosti_norm_maxangle;
                    }
                    Eigen::Vector2d host0_angle = UtiliLine::Normvector2Angle(host0_norm);
                    Eigen::Vector2d hosti_angle = UtiliLine::Normvector2Angle(hosti_norm_maxangle);
                    Para_Line_Feature[ml_id][0] = host0_angle(0);
                    Para_Line_Feature[ml_id][1] = host0_angle(1);
                    Para_Line_Feature[ml_id][2] = hosti_angle(0);
                    Para_Line_Feature[ml_id][3] = hosti_angle(1);
                    problem.AddParameterBlock(Para_Line_Feature[ml_id], 4);
                }

                // std::cout<<"ssdsdfda"<<std::endl;

                // add camera pose
                double Para_Pose[opti_para.kfs.size()][7];
                for (auto kf = opti_para.kfs.begin(); kf != opti_para.kfs.end(); kf++)
                {
                    int kf_id = kf->first;
                    Eigen::Matrix4d Twc = kf->second;
                    Eigen::Matrix3d Rot = Twc.block(0, 0, 3, 3);
                    Eigen::Vector3d Tran = Twc.block(0, 3, 3, 1);
                    Eigen::Quaterniond qua(Rot);

                    Para_Pose[kf_id][0] = Tran(0);
                    Para_Pose[kf_id][1] = Tran(1);
                    Para_Pose[kf_id][2] = Tran(2);
                    Para_Pose[kf_id][3] = qua.x();
                    Para_Pose[kf_id][4] = qua.y();
                    Para_Pose[kf_id][5] = qua.z();
                    Para_Pose[kf_id][6] = qua.w();

                    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
                    problem.AddParameterBlock(Para_Pose[kf_id], 7, local_parameterization);
                    if (kf_id == 0)
                        problem.SetParameterBlockConstant(Para_Pose[kf_id]);
                }

                // construct ceres problem
                for (auto asso_mp_mea = opti_para.asso_mp_meas.begin(); asso_mp_mea != opti_para.asso_mp_meas.end(); asso_mp_mea++)
                {
                    int mp_id = asso_mp_mea->first;
                    if (opti_para.mappoints[mp_id] == Vec3::Zero())
                        continue;
                    if (asso_mp_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;

                    for (int i = 0; i < asso_mp_mea->second.size(); i++)
                    {
                        int kf_id_i = asso_mp_mea->second[i].first;
                        Eigen::Vector2d mp_obser_i = asso_mp_mea->second[i].second;
                        Eigen::Vector2d mp_obser_i_unit((mp_obser_i(0) - cx) / fx, (mp_obser_i(1) - cy) / fy);

                        PointProjectionFactor *f = new PointProjectionFactor(mp_obser_i_unit);
                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Para_Point_Feature[mp_id]);
                    }
                }

                // construct ceres problem
                for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                {
                    int ml_id = asso_ml_mea->first;
                    if (opti_para.maplines[ml_id] == Mat32::Zero())
                        assert(1 == 0);
                    if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;
                    if(Para_Line_Feature_Host_id[ml_id][1] == 0)
                        continue;
                    
                    for (int i = 0; i < asso_ml_mea->second.size(); i++)
                    {
                        if (asso_ml_mea->second[i].first == Para_Line_Feature_Host_id[ml_id][0] || asso_ml_mea->second[i].first == Para_Line_Feature_Host_id[ml_id][1])
                            continue;
                        int kf_id_i = asso_ml_mea->second[i].first;
                        Eigen::Vector4d ml_obser_i = asso_ml_mea->second[i].second;

                        Eigen::Vector4d mp_obser_i_unit(
                            (ml_obser_i(0) - cx) / fx,
                            (ml_obser_i(1) - cy) / fy,
                            (ml_obser_i(2) - cx) / fx,
                            (ml_obser_i(3) - cy) / fy);

                        int kf_id_0 = Para_Line_Feature_Host_id[ml_id][0];
                        int kf_id_1 = Para_Line_Feature_Host_id[ml_id][1];

                        ZhaoLiangLineProjectionFactor *f = new ZhaoLiangLineProjectionFactor(mp_obser_i_unit);
                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Para_Pose[kf_id_0], Para_Pose[kf_id_1], Para_Line_Feature[ml_id]);
                        
                        //Eigen::Matrix2d sqrt_info;
                        //sqrt_info.setIdentity();
                        //ceres::CostFunction* cost_function = ZhaoLiangLineProjectionFactorAutoDiff::Create(mp_obser_i_unit, sqrt_info);
                        //problem.AddResidualBlock(cost_function, loss_function, Para_Pose[kf_id_i], Para_Pose[kf_id_0], Para_Pose[kf_id_1], Para_Line_Feature[ml_id]);
                    }
                }
                // slove ceres problem
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::SPARSE_SCHUR;
                options.max_num_iterations = 100;
                options.minimizer_progress_to_stdout = true;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
#if __VERBOSE__
                std::cout << summary.FullReport() << std::endl;
#endif

                // update ceres results
                for (auto kf = opti_para.kfs.begin(); kf != opti_para.kfs.end(); kf++)
                {
                    int kf_id = kf->first;
                    Eigen::Vector3d Tran(Para_Pose[kf_id][0], Para_Pose[kf_id][1], Para_Pose[kf_id][2]);
                    Eigen::Quaterniond qua(Para_Pose[kf_id][6], Para_Pose[kf_id][3], Para_Pose[kf_id][4], Para_Pose[kf_id][5]);
                    Eigen::Matrix3d Rot = qua.toRotationMatrix();
                    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
                    Twc.block(0, 0, 3, 3) = Rot;
                    Twc.block(0, 3, 3, 1) = Tran;

                    // std::cout<<"pose: "<< kf->second<<std::endl<<Twc<<std::endl;
                    //  opti_para.kfs[kf_id] = Twc;
                    optimized_poses[kf_id] = Twc;
                }

                // update mappoints
                for (auto mp = opti_para.mappoints.begin(); mp != opti_para.mappoints.end(); mp++)
                {
                    int mp_id = mp->first;
                    Vec3 optimized_mappoint;
                    optimized_mappoint << Para_Point_Feature[mp_id][0], //= mp->second(0);
                        Para_Point_Feature[mp_id][1],                   //= mp->second(1);
                        Para_Point_Feature[mp_id][2];                   //= mp->second(2);
                    optimized_mappoints[mp_id] = optimized_mappoint;
                    // std::cout<<"original: "<< mp->second(0)<<","<< mp->second(1)<< ","<< mp->second(2)<<std::endl;
                    // std::cout << "optimized: " << optimized_mappoint(0) << "," << optimized_mappoint(1) << "," << optimized_mappoint(2) << std::endl;
                }

                // update maplines
                // for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                // {
                //     //

                //     int ml_id = asso_ml_mea->first;
                //     if (opti_para.maplines[ml_id] == Mat32::Zero())
                //         assert(1 == 0);
                //     if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                //         continue;

                //     // std::cout << "optimized_maplines_:" << std::endl;

                //     Eigen::Matrix<double, 6, 1> optimized_mapline;
                //     optimized_mapline << Para_Line_Feature[ml_id][0],
                //         Para_Line_Feature[ml_id][1],
                //         Para_Line_Feature[ml_id][2],
                //         Para_Line_Feature[ml_id][3],
                //         Para_Line_Feature[ml_id][4],
                //         Para_Line_Feature[ml_id][5];

                //     // Eigen::Vector3d endpoint_s = opti_para.maplines[ml_id].col(0);
                //     // Eigen::Vector3d endpoint_e = opti_para.maplines[ml_id].col(1);

                //     // Eigen::Matrix<double, 6, 1>
                //     //     opti_ml_plk = UtiliLine::orth_to_plk(optimized_mapline_orth);

                //     Vector6d endpoints = UtiliLine::plk_to_endpoints(optimized_mapline, opti_para.maplines[ml_id]);

                //     optimized_maplines[ml_id].block(0, 0, 3, 1) = endpoints.head(3);
                //     optimized_maplines[ml_id].block(0, 1, 3, 1) = endpoints.tail(3);

                //     // std::cout << "ml_id:" << ml_id << "optimized:" << optimized_maplines[ml_id] << std::endl;
                // }
                return;
            }
        };
    } // namespace optimizer

} // namespace simulator

#endif // __VENOM_SRC_OPTIMIZER_COVISZHAO_BUNDLEADJUSTMENT__
