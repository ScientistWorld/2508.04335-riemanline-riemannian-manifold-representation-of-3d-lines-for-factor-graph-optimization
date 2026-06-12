/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-02-04 16:28:14
 * @LastEditTime: 2023-08-29 13:54:11
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /JointFactorGraph/src/optimizer/CovisPointBA.hpp
 */

#ifndef __VENOM_SRC_OPTIMIZER_COVIS_POINT_BUNDLEADJUSMENT_CERES_HPP__
#define __VENOM_SRC_OPTIMIZER_COVIS_POINT_BUNDLEADJUSMENT_CERES_HPP__

#include "src/manager_env/EnvTrajectory.hpp"
#include <ceres/ceres.h>
// #include <filesystem>
// #include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/optimizer/parametrization/pose_local_parameterization.hpp"
// #include "src/utils/UtilTransformer.hpp"
#include "src/optimizer/factor/LineProjectionFactor.hpp"

namespace simulator
{
    namespace optimizer
    {
        class CovisPointBA
        {
            public:
                // Callback definition
                struct PoseLandmarkIterationCallback : public ceres::IterationCallback
                {
                    std::vector<double *> pose_blocks;     // Each is double[7]
                    std::vector<double *> landmark_blocks; // Each is double[3]

                    std::string save_dir;
                    bool save_to_file;

                    PoseLandmarkIterationCallback(
                        const std::vector<double *> &poses,
                        const std::vector<double *> &landmarks,
                        const std::string &dir = "../iteration_logs/",
                        bool save_file = true)
                        : pose_blocks(poses), landmark_blocks(landmarks),
                          save_dir(dir), save_to_file(save_file)
                    {
                        // if (save_to_file)
                        // {
                        //     std::filesystem::create_directories(save_dir);
                        // }
                    }

                    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override
                    {
                        int iter = summary.iteration;

                        if (!save_to_file)
                            return ceres::SOLVER_CONTINUE;

                        // === Save camera poses ===
                        std::ofstream pose_file(save_dir + "/p_poses_iter_" + std::to_string(iter) + ".txt");
                        for (size_t i = 0; i < pose_blocks.size(); ++i)
                        {
                            double *pose = pose_blocks[i];
                            Eigen::Vector3d t(pose[0], pose[1], pose[2]);
                            Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]);

                            Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
                            Twc.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
                            Twc.block<3, 1>(0, 3) = t;

                            pose_file << Twc << "\n\n";
                        }
                        pose_file.close();

                        // === Save 3D landmarks ===
                        std::ofstream lm_file(save_dir + "/p_points_iter_" + std::to_string(iter) + ".txt");
                        for (size_t i = 0; i < landmark_blocks.size(); ++i)
                        {
                            double *point = landmark_blocks[i];
                            lm_file << point[0] << " " << point[1] << " " << point[2] << "\n";
                        }
                        lm_file.close();

                        return ceres::SOLVER_CONTINUE;
                    }
                };

            static void optimizer(
                simulator::Trajectory *ptr_robot_trajectory,
                simulator::OptimizationManager &opti_para,
                std::map<int, Eigen::Vector3d> &optimized_mappoints,
                std::map<int, Eigen::Matrix4d> &optimized_poses)
            {
                // ==== set parameters ====
                constexpr size_t first_kf_id_in_map = 0;
                constexpr int opmization_times = 1;
                constexpr int max_iters = 5;
                constexpr int min_obser_num = 3;
                const float huber_thres = 1.5;
                const float pixel_sigma = 1.5;

                // ==== intrinsic ====
                double fx; double fy; double cx; double cy;
                fx = ptr_robot_trajectory->cam_intri.fx;
                fy = ptr_robot_trajectory->cam_intri.fy;
                cx = ptr_robot_trajectory->cam_intri.cx;
                cy = ptr_robot_trajectory->cam_intri.cy;

                // ==== construct ceres optim parameters ====
                ceres::Problem problem;
                ceres::LossFunction *loss_function;
                ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering();
                loss_function = new ceres::CauchyLoss(3.0 / fx);

                // ====== 1. add mappoints ======
                // double Para_Point_Landmark[opti_para.mappoints.size()][3];
                // auto iteration = opti_para.mappoints.rbegin();
                // double Para_Point_Landmark[iteration->first + 1][3];
                std::map<int, double[3]> Para_Point_Landmark;
                for (const auto &[mp_id, point] : opti_para.mappoints)
                {
                    if (point.norm() < 0.01)
                        continue;

                    Para_Point_Landmark[mp_id][0] = point.x();
                    Para_Point_Landmark[mp_id][1] = point.y();
                    Para_Point_Landmark[mp_id][2] = point.z();
                    problem.AddParameterBlock(Para_Point_Landmark[mp_id], 3);
                }

                // for (auto mp = opti_para.mappoints.begin(); mp != opti_para.mappoints.end(); mp++)
                // {
                //     int mp_id = mp->first;

                //     // if (opti_para.mappoints[mp_id] == Vec3::Zero())
                //     //     continue;
                //     double point_distance = opti_para.mappoints[mp_id].norm();
                //     if (point_distance < 0.01)
                //         assert(0 == 1);
                //     Para_Point_Landmark[mp_id][0] = mp->second(0);
                //     Para_Point_Landmark[mp_id][1] = mp->second(1);
                //     Para_Point_Landmark[mp_id][2] = mp->second(2);
                //     // std::cout << "id:" << mp_id << "," << mp->second(0) << "," << mp->second(1) << std::endl;
                //     problem.AddParameterBlock(Para_Point_Landmark[mp_id], 3);
                //     // ordering
                // }

                // ====== 2. add camera pose ======
                double Para_Pose[opti_para.kfs.size()][7];
                for (auto kf = opti_para.kfs.begin(); kf != opti_para.kfs.end(); kf++)
                {
                    int kf_id = kf->first;
                    Eigen::Matrix4d Twc = kf->second;
                    Eigen::Matrix3d Rot = Twc.block(0,0,3,3);
                    Eigen::Vector3d Tran = Twc.block(0,3,3,1);
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

                // ====== 3. add reprojection residuals ======
                for(auto asso_mp_mea = opti_para.asso_mp_meas.begin(); asso_mp_mea !=  opti_para.asso_mp_meas.end(); asso_mp_mea++)
                {
                    // std::map<int /*mappoint_id*/, std::vector<std::pair<int /*frame_id*/, Eigen::Vector2d /*2d_meas*/>>> asso_mp_meas;
                    // [mp_id, observation]
                    int mp_id = asso_mp_mea->first;
                    if(opti_para.mappoints[mp_id]==Vec3::Zero())
                        assert(1 == 0);

                    if(asso_mp_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;

                    for(int i = 0; i < asso_mp_mea->second.size(); i++)
                    {
                        int kf_id_i = asso_mp_mea->second[i].first;
                        Eigen::Vector2d mp_obser_i = asso_mp_mea->second[i].second;
                        
                        Eigen::Vector2d mp_obser_i_unit((mp_obser_i(0)-cx) / fx, (mp_obser_i(1)-cy) / fy);

                        // ceres::CostFunction* f = new ceres::NumericDiffCostFunction<PointProjectionFactor, ceres::CENTRAL, 2, 7, 3>(
                        //     new PointProjectionFactor(mp_obser_i_unit));
                        PointProjectionFactor *f = new PointProjectionFactor(mp_obser_i_unit);
                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Para_Point_Landmark[mp_id]);
                    }
                }

                // slove ceres problem
                ceres::Solver::Options options;
                options.minimizer_progress_to_stdout = true;
                // options.max_num_iterations = 15;
                // options.linear_solver_type = ceres::DENSE_SCHUR;
                options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.update_state_every_iteration = true;
                options.gradient_tolerance = 1e-6;
                options.function_tolerance = 1e-7;
                options.max_num_iterations = 100;

                std::vector<double *> pose_blocks;
                std::vector<double *> landmark_blocks;

                // collect all pose pointers
                for (auto &kv : opti_para.kfs)
                {
                    int kf_id = kv.first;
                    pose_blocks.push_back(Para_Pose[kf_id]); // Para_Pose[kf_id] is double[7]
                }

                // collect all landmark pointers
                for (auto &mp : opti_para.mappoints)
                {
                    int mp_id = mp.first;
                    landmark_blocks.push_back(Para_Point_Landmark[mp_id]); // double[3]
                }

                // create and attach callback
                auto *callback = new PoseLandmarkIterationCallback(pose_blocks, landmark_blocks, "../iteration_logs", true);
                options.update_state_every_iteration = true;
                options.callbacks.push_back(callback);

                ceres::Solver::Summary summary;
                ceres::Solve (options, &problem, & summary);

#if __VERBOSE__
                std::cout
                    << summary.FullReport() << std::endl;
#endif
                // update ceres results
                for (auto kf =  opti_para.kfs.begin(); kf !=  opti_para.kfs.end(); kf++)
                {
                    int kf_id = kf->first;
                    Eigen::Vector3d tran(Para_Pose[kf_id][0], Para_Pose[kf_id][1], Para_Pose[kf_id][2]);
                    Eigen::Quaterniond qua(Para_Pose[kf_id][6], Para_Pose[kf_id][3], Para_Pose[kf_id][4], Para_Pose[kf_id][5]);
                    Eigen::Matrix3d Rot = qua.toRotationMatrix();
                    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
                    Twc.block(0,0,3,3) = Rot;
                    Twc.block(0, 3, 3, 1) = tran;
                    // kfs[kf_id] = Twc;
                    optimized_poses[kf_id]=Twc;
                }

                // update mappoints
                for (auto mp = opti_para.mappoints.begin(); mp != opti_para.mappoints.end(); mp++)
                {
                    int mp_id = mp->first;
                    if(opti_para.mappoints[mp_id]==Vec3::Zero())
                        continue;
                    Vec3 optimized_mappoint;
                    optimized_mappoint << Para_Point_Landmark[mp_id][0], //= mp->second(0);
                        Para_Point_Landmark[mp_id][1],                   //= mp->second(1);
                        Para_Point_Landmark[mp_id][2];                   //= mp->second(2);
                    optimized_mappoints[mp_id] = optimized_mappoint;
                }
                return;
            }
        };
    } // namespace optimizer
} // namespace simulator

#endif // __VENOM_SRC_OPTIMIZER_COVIS_POINT_BUNDLEADJUSMENT_CERES_HPP__
