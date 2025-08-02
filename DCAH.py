def DCAH(KL_loss, CAQ_loss):
  return KL_loss + CAQ_loss
    
def KL_loss():
    # Normalize features with L2 normalization
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrix for teacher features
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    # Compute cross-similarity matrix between student and teacher features
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    teacher_topk_values, teacher_topk_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    topk_gallery_indices = teacher_topk_indices[:, :top_k]

    student_topk_scores = torch.gather(cross_similarity, 1, topk_gallery_indices)
    teacher_topk_scores = teacher_topk_values[:, :top_k]

    # Compute probability distributions for student and teacher
    student_probs = F.log_softmax(student_topk_scores / temp_student, dim=1)
    teacher_probs = F.softmax(teacher_topk_scores / temp_teacher, dim=1)

    # Compute KL divergence loss
    loss = F.kl_div(student_probs, teacher_probs.detach(), reduction='sum') / teacher_features.shape[0]

    return loss
    
def CAQ_loss(student_features, quantification_teacher_features, hash_dim, bata=0.01): 

    batch_size = student_features.shape[0]

    quantification_student_features = torch.sign(student_features) 
    quantification_student_features = quantification_student_features + (student_features - student_features.detach())  
    
    # Normalize student and teacher features
    student_features = F.normalize(student_features, p=2, dim=1)
    quantification_teacher_features = F.normalize(quantification_teacher_features, p=2, dim=1)
    quantification_student_features = F.normalize(quantification_student_features, p=2, dim=1)
    
    # Compute similarity matrices <quantification_student_features-teacher_features 
    teacher_similarity = student_features.double().mm(quantification_teacher_features.double().t())
    cross_similarity = quantification_student_features.double().mm(quantification_teacher_features.double().t())
  
    q_loss = torch.sum(torch.abs(teacher_similarity - cross_similarity)) / (batch_size * batch_size)

    t = {
         16:(0.5,0.1),
         32:(0.5,0.05),
         48:(0.5,0.05),
         64:(0.5,0.005),
        }
    prob_Q_S = F.softmax(cross_similarity / t[hash_dim][0], dim=-1)
    prob_S = F.softmax(teacher_similarity / t[hash_dim][1], dim=-1)
    
    loss_ca = F.kl_div(
        input=prob_Q_S.log(), 
        target=prob_S.detach(),       
        reduction="batchmean", 
        log_target=False
    )
    
    return loss_ca + bata * q_loss
