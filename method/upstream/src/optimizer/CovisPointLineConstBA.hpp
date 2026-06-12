#ifndef __VENOM_SRC_OPTIMIZER_C_POINT_LINE_CONSTR_BUNDLEADJUSTMENT__
#define __VENOM_SRC_OPTIMIZER_C_POINT_LINE_CONSTR_BUNDLEADJUSTMENT__

#include <ceres/ceres.h>

#include "src/manager_env/EnvTrajectory.hpp"
#include "src/optimizer/factor/LineProjectionFactor.hpp"
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/optimizer/parametrization/pose_local_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"

namespace simulator
{
    namespace optimizer
    {
        class CovisPointLineConstrBA
        {
        public:
            struct PoseLandmarkIterationCallback : public ceres::IterationCallback
            {
                std::map<int, double[3]> &points;
                std::map<int, Eigen::Vector4d> &lines;
                double (*poses)[7];
                std::map<int, Eigen::Matrix<double, 3, 2>> &ref_maplines;
                std::string save_path;
                int iteration = 0;

                PoseLandmarkIterationCallback(std::map<int, double[3]> &pts,
                                              std::map<int, Eigen::Vector4d> &lns,
                                              double (*p)[7],
                                              std::map<int, Eigen::Matrix<double, 3, 2>> &ref_ml,
                                              const std::string &path = "../iteration_logs/")
                    : points(pts), lines(lns), poses(p), ref_maplines(ref_ml), save_path(path) {}

                virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary)
                {
                    const int iter = summary.iteration;

                    // Save points
                    std::ofstream pt_file(save_path + "/pl_points_iter_" + std::to_string(iter) + ".txt");
                    for (const auto &[id, pt] : points)
                    {
                        pt_file << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
                    }
                    pt_file.close();

                    // Save lines as endpoints
                    std::ofstream line_file(save_path + "/pl_lines_iter_" + std::to_string(iter) + ".txt");
                    for (const auto &[id, orth] : lines)
                    {
                        // Only convert if we have reference endpoints
                        if (ref_maplines.find(id) == ref_maplines.end())
                            continue;

                        Eigen::Matrix<double, 6, 1> plk = UtiliLine::orth_to_plk(orth);
                        Vector6d line_endpoints = UtiliLine::plk_to_endpoints(plk, ref_maplines[id]);
                        line_file << line_endpoints.transpose() << "\n"; // x1 y1 z1 x2 y2 z2
                    }
                    line_file.close();

                    // Save poses
                    std::ofstream pose_file(save_path + "/pl_poses_iter_" + std::to_string(iter) + ".txt");
                    for (int i = 0; i < 500; ++i)
                    {
                        if (poses[i][0] == 0 && poses[i][1] == 0 && poses[i][2] == 0)
                            continue;
                        pose_file << poses[i][0] << " " << poses[i][1] << " " << poses[i][2] << " "
                                  << poses[i][3] << " " << poses[i][4] << " " << poses[i][5] << " " << poses[i][6] << "\n";
                    }
                    pose_file.close();

                    iteration++;
                    return ceres::SOLVER_CONTINUE;
                }
            };

            static void optimizer(
                simulator::Trajectory *ptr_robot_trajectory,
                simulator::OptimizationManager &opti_para,
                std::map<int, Eigen::Vector3d> &optimized_mappoints,
                std::map<int, Eigen::Matrix<double, 3, 2>> &optimized_maplines,
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
                double fx, fy, cx, cy;
                fx = ptr_robot_trajectory->cam_intri.fx;
                fy = ptr_robot_trajectory->cam_intri.fy;
                cx = ptr_robot_trajectory->cam_intri.cx;
                cy = ptr_robot_trajectory->cam_intri.cy;

                // ==== construct ceres optim parameters ====
                ceres::Problem problem;
                ceres::LossFunction *loss_function;
                ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering();
                loss_function = new ceres::CauchyLoss(3.0 / fx);

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

                // ====== 2. add maplines ======
                // double Param_Line_Feature[opti_para.maplines.size()][4];
                // std::map<int, double[3]> Para_Line_Landmark;
                std::map<int, Eigen::Vector4d> Param_Line_Feature;

                for (const auto &ml : opti_para.maplines)
                {
                    int ml_id = ml.first;
                    const Mat32 &line_endpoints = ml.second;

                    // Skip uninitialized lines
                    if (line_endpoints == Mat32::Zero())
                        continue;

                    // Check for invalid or too short line
                    double line_length = (line_endpoints.col(0) - line_endpoints.col(1)).norm();
                    if (line_length < 0.01)
                    {
                        std::cerr << "[Warning] Line " << ml_id << " too short.\n";
                        continue;
                        // parallel_line_groups
                    }

                    // Copy endpoints
                    Eigen::Vector3d start = line_endpoints.col(0);
                    Eigen::Vector3d end = line_endpoints.col(1);

                    // Try to find which group this line belongs to
                    for (const auto &group : opti_para.asso_paralineid_mlids)
                    {
                        const auto &para_lines_id = group.second;

                        // If this line is in the group
                        if (std::find(para_lines_id.begin(), para_lines_id.end(), ml_id) != para_lines_id.end())
                        {
                            // Reference line's direction
                            int ref_ml_id = para_lines_id[0];
                            if (opti_para.maplines.count(ref_ml_id) == 0)
                                break;

                            Eigen::Vector3d ref_start = opti_para.maplines[ref_ml_id].col(0);
                            Eigen::Vector3d ref_end = opti_para.maplines[ref_ml_id].col(1);
                            Eigen::Vector3d ref_dir = (ref_end - ref_start).normalized();

                            // Current line direction
                            Eigen::Vector3d cur_dir = (end - start).normalized();

                            // Flip direction if inconsistent
                            if (ref_dir.dot(cur_dir) < 0)
                            {
                                std::swap(start, end); // flip
                            }

                            break; // only one group contains this line
                        }
                    }

                    // Compute PLK representation
                    // const Eigen::Vector3d &start = line_endpoints.col(0);
                    // const Eigen::Vector3d &end = line_endpoints.col(1);
                    Eigen::Vector3d direction = (end - start).normalized();
                    Eigen::Vector3d normal = start.cross(end).normalized();
                    double d = (start.cross(end)).norm() / line_length;
                    Eigen::Matrix<double, 6, 1> line_plk;
                    line_plk << d * normal, direction;

                    // Convert to orthonormal representation
                    Eigen::Vector4d line_orth = UtiliLine::plk_to_orth(line_plk);
                    Param_Line_Feature[ml_id] = line_orth;

#if __VERBOSE__OFF
                    // Debug: Verify PLK <-> Orth consistency
                    Eigen::Matrix<double, 6, 1> plk_back = UtiliLine::orth_to_plk(line_orth);
                    Vector3d n = plk_back.head(3);
                    Vector3d v = plk_back.tail(3);
                    std::cout << "PLK from orth: n = " << n.transpose()
                              << ", v = " << v.transpose()
                              << ", |n|/|v| = " << n.norm() / v.norm() << std::endl;
#endif

                    // Add parameter block to Ceres
                    ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
                    problem.AddParameterBlock(Param_Line_Feature[ml_id].data(), 4, local_parameterization_line);
                }

                //  ====== 3. add camera pose  ======
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

                // construct ceres problem (point reprojection factor)
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
                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Para_Point_Landmark[mp_id]);
                    }
                }

                // construct ceres problem (line reprojection factor)
                for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                {
                    int ml_id = asso_ml_mea->first;
                    if (opti_para.maplines[ml_id] == Mat32::Zero())
                        assert(1 == 0);
                    if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;

                    for (int i = 0; i < asso_ml_mea->second.size(); i++)
                    {
                        int kf_id_i = asso_ml_mea->second[i].first;
                        Eigen::Vector4d ml_obser_i = asso_ml_mea->second[i].second;

                        Eigen::Vector4d mp_obser_i_unit(
                            (ml_obser_i(0) - cx) / fx,
                            (ml_obser_i(1) - cy) / fy,
                            (ml_obser_i(2) - cx) / fx,
                            (ml_obser_i(3) - cy) / fy);

                        LineProjectionFactor *f = new LineProjectionFactor(mp_obser_i_unit);
                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Param_Line_Feature[ml_id].data());
                    }
                }

                // construct ceres problem (parallel-line constraints
                for (const auto &group : opti_para.asso_paralineid_mlids)
                {
                    const auto &para_lines_id = group.second; // vector<int /*paramapline_id*/> of ml_ids

                    for (size_t i = 0; i < para_lines_id.size(); ++i)
                    {
                        for (size_t j = i + 1; j < para_lines_id.size(); ++j)
                        {
                            int ml_id_i = para_lines_id[i];
                            int ml_id_j = para_lines_id[j];

                            // Skip uninitialized
                            if (Param_Line_Feature.count(ml_id_i) == 0 ||
                                Param_Line_Feature.count(ml_id_j) == 0)
                                continue;

                            Eigen::Vector3d dir_i = UtiliLine::orth_to_plk(Param_Line_Feature[ml_id_i]).tail<3>().normalized();
                            Eigen::Vector3d dir_j = UtiliLine::orth_to_plk(Param_Line_Feature[ml_id_j]).tail<3>().normalized();

                            double dot_product = dir_i.dot(dir_j);

                            // Only add constraint if they are close to parallel (e.g., > cos(10 deg) ≈ 0.9848)
                            if (dot_product > 0.92)
                            {
                                // std::cout << "add para constraint between lines " << ml_id_i << " and " << ml_id_j
                                //           << " (dot=" << dot_product << ")" << std::endl;

                                ceres::CostFunction *f = new ParallelLineFactor();
                                problem.AddResidualBlock(f, loss_function,
                                                         Param_Line_Feature[ml_id_i].data(),
                                                         Param_Line_Feature[ml_id_j].data());
                            }
                            else
                            {
                                std::cout << "skip inconsistent pair: " << ml_id_i << " and " << ml_id_j
                                          << " (dot=" << dot_product << ")" << std::endl;
                            }

                            // std::cout << "check para:" << opti_para.maplines[ml_id_i] << "," << opti_para.maplines[ml_id_j] << std::endl;
                            // ceres::CostFunction *f = new ParallelLineFactor();
                            // problem.AddResidualBlock(f, loss_function,
                            //                          Param_Line_Feature[ml_id_i].data(),
                            //                          Param_Line_Feature[ml_id_j].data());
                        }
                    }
                }

                // slove ceres problem
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::SPARSE_SCHUR;
                options.max_num_iterations = 100;
                options.gradient_tolerance = 1e-6;
                options.function_tolerance = 1e-7;

                std::vector<double *> pose_blocks;
                std::vector<double *> point_landmark_blocks;
                std::vector<double *> line_landmark_blocks;

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
                    point_landmark_blocks.push_back(Para_Point_Landmark[mp_id]); // double[3]
                }

                // collect all landmark pointers
                for (auto &ml : opti_para.maplines)
                {
                    int ml_id = ml.first;
                    line_landmark_blocks.push_back(Param_Line_Feature[ml_id].data()); // double[3]
                }
                // create and attach callback
                // PoseLandmarkIterationCallback *callback = new PoseLandmarkIterationCallback(Para_Point_Landmark, Param_Line_Feature, Para_Pose, "../iteration_logs");
                // auto *callback = new PoseLandmarkIterationCallback(pose_blocks, landmark_blocks, "../iteration_logs", true);
                auto *callback = new PoseLandmarkIterationCallback(
                    Para_Point_Landmark,
                    Param_Line_Feature,
                    Para_Pose,
                    opti_para.maplines, // pass reference lines
                    "../iteration_logs/");
                options.update_state_every_iteration = true;
                options.callbacks.push_back(callback);

                options.minimizer_progress_to_stdout = true;
                // options.max_num_iterations = 5;
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
                    optimized_mappoint << Para_Point_Landmark[mp_id][0], //= mp->second(0);
                        Para_Point_Landmark[mp_id][1],                   //= mp->second(1);
                        Para_Point_Landmark[mp_id][2];                   //= mp->second(2);
                    optimized_mappoints[mp_id] = optimized_mappoint;
                    // std::cout<<"original: "<< mp->second(0)<<","<< mp->second(1)<< ","<< mp->second(2)<<std::endl;
                    // std::cout << "optimized: " << optimized_mappoint(0) << "," << optimized_mappoint(1) << "," << optimized_mappoint(2) << std::endl;
                }

                // update maplines
                for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                {
                    //
                    int ml_id = asso_ml_mea->first;
                    if (opti_para.maplines[ml_id] == Mat32::Zero())
                        assert(1 == 0);
                    if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;

                    // std::cout << "optimized_maplines_:" << std::endl;
                    Eigen::Vector4d optimized_mapline_orth;
                    optimized_mapline_orth << Param_Line_Feature[ml_id][0],
                        Param_Line_Feature[ml_id][1],
                        Param_Line_Feature[ml_id][2],
                        Param_Line_Feature[ml_id][3];

                    Eigen::Matrix<double, 6, 1>
                        opti_ml_plk = UtiliLine::orth_to_plk(optimized_mapline_orth);

                    Vector6d EndPoints3d = UtiliLine::plk_to_endpoints(opti_ml_plk, opti_para.maplines[ml_id]);

                    // EndPoints3d = EndPoints3d / obser_num;
                    optimized_maplines[ml_id].block(0, 0, 3, 1) = EndPoints3d.head(3);
                    optimized_maplines[ml_id].block(0, 1, 3, 1) = EndPoints3d.tail(3);

                    // Eigen::Matrix<double, 6, 1>
                    //     opti_ml_plk = UtiliLine::orth_to_plk(optimized_mapline_orth);
                    // Vector6d endpoints = UtiliLine::plk_to_endpoints(opti_ml_plk, opti_para.maplines[ml_id]);

                    // optimized_maplines[ml_id].block(0, 0, 3, 1) = endpoints.head(3);
                    // optimized_maplines[ml_id].block(0, 1, 3, 1) = endpoints.tail(3);

                    // std::cout << "ml_id:" << ml_id << "optimized:" << optimized_maplines[ml_id] << std::endl;
                }
                return;
            }
        };
    } // namespace optimizer

} // namespace simulator

#endif // __VENOM_SRC_OPTIMIZER_C_POINT_LINE_CONSTR_BUNDLEADJUSTMENT__